import click
import time

import uvicorn

import api
from camera import Camera
from simulator import Simulator
from gui import WebGuiConnector
from api import api
import multiprocessing

@click.command()
@click.option('--timing', default=10, help='Numerical speed of the simulation. 1 -> real time, 2 -> 2x real time, ... 0 -> as fast as possible.')
@click.option('--time-step', default=20, help='Time step in seconds for the STK animation. Only applicable if simulator is stk, ignored otherwise.')

def main(timing, time_step):
    manager = multiprocessing.Manager()
    shared_data_dict = manager.dict()
    shared_data_dict["satellite_position"] = (0.0, 0.0, 0.0)  # (lon, lat, alt)

    sim_proc = multiprocessing.Process(
        target=run_sim,
        args=(shared_data_dict, timing, time_step)
    )

    api_proc = multiprocessing.Process(
        target=run_api,
        args=(shared_data_dict,)
    )

    sim_proc.start()
    api_proc.start()

    print("Both processes are running. Press Ctrl+C to stop.")

    try:
        sim_proc.join()
        api_proc.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sim_proc.terminate()
        api_proc.terminate()

def run_api(shared_data_dict):
    api.state.shared_data = shared_data_dict
    uvicorn.run(api, host="0.0.0.0", port=8000)

def run_sim(shared_data_dict, timing, time_step):
    # Initialize the simulation GUI connector
    gui = WebGuiConnector()

    # Initialize the simulation engine
    line1 = "1 60989U 24157A   26075.16558042  .00000129  00000-0  65710-4 0  9997"
    line2 = "2 60989  98.5677 151.2852 0000884 109.8893 250.2385 14.30816791 79683"
    sim_engine = Simulator("SatelliteName", TLE=[line1, line2], t0=None, timing_mode=timing, time_step=time_step)

    # Add subsystems
    camera = Camera(shared_data_dict=shared_data_dict)

    # PRISM loop — ticks in the sim process to track entropy drift over time
    try:
        from haic.prism_loop import get_prism_loop
        prism_loop = get_prism_loop()
        prism_enabled = True
    except Exception as e:
        print(f"[HAIC] PRISM loop unavailable: {e}")
        prism_loop = None
        prism_enabled = False

    # Run the simulation
    sim_engine.reset()

    tick_counter = 0
    PRISM_TICK_INTERVAL = 100  # tick PRISM every 100 sim steps (~10s at 0.1s sleep)

    while True:
        sim_engine.sim_step()
        time.sleep(0.1)

        tick_counter += 1
        if prism_enabled and prism_loop and tick_counter % PRISM_TICK_INTERVAL == 0:
            prism_loop.tick()

if __name__ == '__main__':
    main()
