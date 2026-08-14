const API_BASE = "http://127.0.0.1:8000";

function canvasId(sourceVideo, scene) {
  return `chart-${sourceVideo}-${scene}`.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function renderScene(sourceVideo, scene) {
  const card = document.createElement("div");
  card.className = "scene-card";

  const title = document.createElement("h3");
  title.innerHTML =
    `Scene ${scene.scene} <span>${scene.start.toFixed(2)}s&ndash;${scene.end.toFixed(2)}s</span>`;
  const badge = document.createElement("span");
  badge.className = `badge ${scene.has_embedding ? "on" : ""}`;
  badge.textContent = scene.has_embedding ? "embedded" : "no embedding";
  title.appendChild(badge);
  card.appendChild(title);

  const thumbs = document.createElement("div");
  thumbs.className = "thumbs";
  for (const thumb of scene.thumbnails) {
    if (!thumb) continue;
    const img = document.createElement("img");
    img.src = `${API_BASE}${thumb}`;
    img.loading = "lazy";
    thumbs.appendChild(img);
  }
  card.appendChild(thumbs);

  if (scene.rd_curve) {
    const canvas = document.createElement("canvas");
    canvas.id = canvasId(sourceVideo, scene.scene);
    card.appendChild(canvas);
    new Chart(canvas, {
      type: "line",
      data: {
        labels: scene.rd_curve.kbps,
        datasets: [
          {
            label: "VMAF vs kbps",
            data: scene.rd_curve.vmaf,
          },
        ],
      },
      options: { maintainAspectRatio: false },
    });
  }

  return card;
}

function renderVideo(video) {
  const section = document.createElement("section");

  const heading = document.createElement("h2");
  heading.textContent = video.source_video;
  section.appendChild(heading);

  const scenes = document.createElement("div");
  scenes.className = "scenes";
  for (const scene of video.scenes) {
    scenes.appendChild(renderScene(video.source_video, scene));
  }
  section.appendChild(scenes);

  return section;
}

fetch(`${API_BASE}/videos`)
  .then((res) => res.json())
  .then((videos) => {
    const container = document.getElementById("videos");
    if (videos.length === 0) {
      container.textContent = "No videos ingested yet.";
      container.className = "empty";
      return;
    }
    for (const video of videos) {
      container.appendChild(renderVideo(video));
    }
  });
