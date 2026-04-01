/**
 * StimulusViewer — shows satellite images from a GroundingStimulus side-by-side.
 *
 * Lazily fetches each image by index from the /haic/stimulus/:id/image/:index
 * endpoint and renders them with metadata overlays.
 */
import React, { useEffect, useState } from "react";
import { fetchStimulusImageDataUrl, type GroundingStimulus } from "./haicApi";

interface Props {
  stimulus: GroundingStimulus;
}

export const StimulusViewer: React.FC<Props> = ({ stimulus }) => {
  const [dataUrls, setDataUrls] = useState<(string | null)[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);

  const availableImages = stimulus.images.filter((img) => img.has_image);

  useEffect(() => {
    if (!availableImages.length) return;

    // Resolve indices in the original images array for available images
    const availableIndices = stimulus.images
      .map((img, i) => ({ img, i }))
      .filter(({ img }) => img.has_image)
      .map(({ i }) => i);

    const urls: (string | null)[] = new Array(stimulus.images.length).fill(null);
    setDataUrls([...urls]);

    availableIndices.forEach(async (origIdx) => {
      const dataUrl = await fetchStimulusImageDataUrl(stimulus.stimulus_id, origIdx);
      setDataUrls((prev) => {
        const next = [...prev];
        next[origIdx] = dataUrl;
        return next;
      });
    });
  }, [stimulus.stimulus_id]);

  if (!availableImages.length) {
    return (
      <div className="stim-viewer stim-no-image">
        <span>No imagery available for this position</span>
      </div>
    );
  }

  const activeImage = stimulus.images[activeIdx];
  const activeUrl = dataUrls[activeIdx];
  const meta = activeImage?.metadata;

  return (
    <div className="stim-viewer">
      {/* Tab bar */}
      <div className="stim-tabs">
        {stimulus.images.map((img, i) => (
          img.has_image && (
            <button
              key={i}
              className={`stim-tab ${activeIdx === i ? "active" : ""}`}
              onClick={() => setActiveIdx(i)}
            >
              {labelForType(img.stimulus_type)}
            </button>
          )
        ))}
      </div>

      {/* Image */}
      <div className="stim-image-container">
        {activeUrl ? (
          <img
            src={activeUrl}
            alt={`Satellite: ${activeImage?.stimulus_type}`}
            className="stim-image"
          />
        ) : (
          <div className="stim-loading">
            <div className="stim-spinner" />
            <span>Loading…</span>
          </div>
        )}

        {/* Metadata overlay */}
        {meta && (
          <div className="stim-meta-overlay">
            {meta.source && <span>{meta.source}</span>}
            {meta.cloud_cover != null && (
              <span>☁ {meta.cloud_cover.toFixed(0)}%</span>
            )}
            {meta.elevation_degrees != null && (
              <span>↑ {meta.elevation_degrees.toFixed(1)}°</span>
            )}
          </div>
        )}
      </div>

      {/* Context */}
      <div className="stim-context">{stimulus.observation_context}</div>
    </div>
  );
};

function labelForType(t: string): string {
  switch (t) {
    case "sentinel_rgb": return "RGB";
    case "sentinel_multispectral": return "Multi";
    case "mapbox_perspective": return "Perspective";
    default: return t;
  }
}
