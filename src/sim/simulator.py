import datetime
import logging
import time

import numpy as np
from pydispatch import dispatcher
from pyorbital.orbital import Orbital

logger = logging.getLogger(__name__)

TOPIC_SIMULATION_COMMAND = "simulation.command"
TOPIC_SATELLITE_GROUND_POSITION = "satellite.ground_position"
TOPIC_SIMULATION_STEP_FORWARD = "simulation.step_forward"
TOPIC_SIMULATION_TICK = "simulation.tick"



class Simulator:
    def __init__(self, name, TLE, t0=None, timing_mode=0, time_step=10):        
        self.name = name
        self.satellite = Orbital(name, line1=TLE[0], line2=TLE[1])

        self.timing_mode = timing_mode  # 0 = as fast as possible, 1 = real time, 2 = 2x real time, etc.
        self.time_step = time_step  # in seconds

        self.sim_t0 = t0
        if self.sim_t0 is None:
            self.sim_t0 = time.time()

        self.reset()

        dispatcher.connect(self.on_command, signal=TOPIC_SIMULATION_COMMAND)

    def _coerce_orbit_time(self, timestamp):
        if isinstance(timestamp, np.datetime64):
            return timestamp.astype("datetime64[ms]")
        if isinstance(timestamp, datetime.datetime):
            dt = timestamp
        elif isinstance(timestamp, (int, float)):
            dt = datetime.datetime.fromtimestamp(float(timestamp), tz=datetime.timezone.utc)
        elif isinstance(timestamp, str):
            normalized = timestamp.strip()
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            dt = datetime.datetime.fromisoformat(normalized)
        else:
            raise TypeError(f"Unsupported orbit timestamp type: {type(timestamp)!r}")

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)
        return np.datetime64(dt.replace(tzinfo=None))

    def get_orbital_location(self, timestamp):
        np_t = self._coerce_orbit_time(timestamp)
        lon, lat, alt = self.satellite.get_lonlatalt(np_t)
        return (lon, lat, alt)
    
    def reset(self):
        self.utcg_time = self.sim_t0
        self.currentTime_EpSec = 0
        self.start_time = None
        self.sim_is_running = False
        self.sim_outstanding_rewind_command = False

    def sim_step(self):
        # tick in any case. this is used to e.g. fetch commands -> needed even if the sim is not running
        self.tick()

        if self.sim_is_running:
            logger.debug("Advancing simulation by %s seconds.", self.time_step)
            if not self.start_time:
                self.start_time = time.time()
            if self.timing_mode: 
                # wait until real time catches up. if timing_mode=0, we run as fast as possible -> this is skipped
                if (time.time() - self.start_time)*self.timing_mode < self.currentTime_EpSec:
                    return False
                
            self.currentTime_EpSec += self.time_step
            self.utcg_time = self.sim_t0 + self.currentTime_EpSec
                
                # This is done for compatibility with other software (non public)
            for i in range(3, -1, -1):
                dispatcher.send(
                    signal=TOPIC_SIMULATION_STEP_FORWARD,
                    sender=str(self),
                    data={'counter': i},
                    time=self.utcg_time,
                    time_epsec=self.currentTime_EpSec,
                )

            self._publish_satellite_ground_position(self.utcg_time, self.currentTime_EpSec)
        if self.sim_outstanding_rewind_command:
            self.sim_outstanding_rewind_command = False
            self.reset()

    def tick(self):
        dispatcher.send(
            signal=TOPIC_SIMULATION_TICK,
            sender=str(self),
            data={},
            time=self.utcg_time,
            time_epsec=self.currentTime_EpSec,
        )

    def _publish_satellite_ground_position(self, utcg_time, sim_time):
        lon, lat, alt = self.get_orbital_location(utcg_time)
        dispatcher.send(
            signal=TOPIC_SATELLITE_GROUND_POSITION,
            sender=str(self),
            data={'lon': lon, 'lat': lat, 'alt': alt},
            time=datetime.datetime.fromtimestamp(utcg_time, tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            time_epsec=sim_time,
        )

    
    def on_command(self, sender, data):
        command = data.get('command', '')
        parameters = data.get('parameters', {})
        logger.info("Command received: %s", command)
        if command == 'start':
            if "start_time" in parameters:
                self.set_start_time(parameters.get("start_time"))
            self.set_sim_speed(step_size=parameters.get('step_size_seconds', 10),
                               replay_speed=parameters.get('replay_speed', 1.0))
            self.sim_is_running = True
        elif command == 'set_start_time':
            self.set_start_time(parameters.get("start_time"))
        elif command == 'set_step_size':
            self.set_sim_speed(step_size=parameters.get('step_size_seconds'))
        elif command == 'set_replay_speed':
            self.set_sim_speed(replay_speed=parameters.get('replay_speed'))
        elif command == 'pause':
            self.sim_is_running = False
        elif command == 'reset':
            self.sim_is_running = False
            self.sim_outstanding_rewind_command = True
        else:
            logger.warning("Unknown command: %s", command)

    def set_sim_speed(self, step_size=None, replay_speed=None):
        if step_size is None:
            pass
        elif step_size > 0:
            self.time_step = step_size
        else:
            self.time_step = 10  # default

        if replay_speed is None:
            pass
        elif replay_speed > 0:
            self.timing_mode = replay_speed
        else:
            self.timing_mode = 0  # as fast as possible

        # correct the start time to account for speed change. Otherwise there will be jumps in the timeline
        self.start_time = time.time() - (self.currentTime_EpSec / self.timing_mode if self.timing_mode > 0 else 0)

    def _parse_start_time(self, start_time):
        if not isinstance(start_time, str):
            return None

        normalized = start_time.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        try:
            dt = datetime.datetime.fromisoformat(normalized)
        except ValueError:
            return None

        # Only accept ISO-8601 timestamps that explicitly represent UTC.
        if dt.tzinfo is None or dt.utcoffset() != datetime.timedelta(0):
            return None

        return dt.timestamp()

    def set_start_time(self, start_time):
        start_ts = self._parse_start_time(start_time)
        if start_ts is None:
            logger.warning("Invalid start_time %r — expected ISO-8601 UTC (e.g. 2026-03-12T12:34:56Z).", start_time)
            return False

        self.sim_t0 = start_ts
        self.utcg_time = self.sim_t0
        self.currentTime_EpSec = 0
        self.start_time = None
        logger.info(
            "Simulation start time set to: %s",
            datetime.datetime.fromtimestamp(self.sim_t0, tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        return True
        
