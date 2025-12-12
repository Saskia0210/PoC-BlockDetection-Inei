import os
import rasterio
import numpy as np
from rasterio.windows import Window
import cv2
import json

def normalize_percentiles(x, pmin=2, pmax=98):
    x = x.astype(np.float32)
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)

    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return (x * 255).astype(np.uint8)


def tile_geotiff(
    tif_path,
    out_dir="tiles/",
    tile_size=512,
    overlap=0.5,
    nodata_threshold=0.999   # porcentaje máximo permitido (30%)
    ,black_tile_threshold=5.0
):
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        bands = src.count
        transform = src.transform
        nodata_value = src.nodata

        stride = int(tile_size * (1 - overlap))

        metadata = {
            "tile_size": tile_size,
            "overlap": overlap,
            "stride": stride,
            "tiles": [],
            "transform": list(transform),
            "crs": str(src.crs),
            "width": W,
            "height": H,
            "nodata_value": nodata_value
        }

        tile_id = 0

        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):

                window = Window(x, y, tile_size, tile_size)
                tile = src.read(window=window)  # (bands, H, W)

                # --- Detectar NoData ---
                if nodata_value is not None:
                    nodata_mask = (tile == nodata_value)
                    nodata_fraction = nodata_mask.sum() / tile.size
                else:
                    nodata_fraction = 0  # No sabemos, asumimos válido

                # Saltar tile si tiene demasiado nodata
                if nodata_fraction > nodata_threshold:
                    continue

                # Usar bandas RGB
                if bands >= 3:
                    tile_rgb = np.stack([tile[0], tile[1], tile[2]], axis=-1)
                else:
                    raise ValueError("TIF must have at least 3 bands (RGB).")

                tile_rgb_8bit = normalize_percentiles(tile_rgb)

                # --- 2. Detect Visually Black Tile ---
                # Check the average intensity across all channels (0-255 scale)
                mean_intensity = np.mean(tile_rgb_8bit)
                
                if mean_intensity < black_tile_threshold:
                    # Skip tile if it is too dark (visually black)
                    print(f"Skipping tile_{tile_id} at ({x}, {y}). Mean intensity ({mean_intensity:.2f}) is below threshold ({black_tile_threshold}).")
                    continue

                tile_path = os.path.join(out_dir, f"tile_{tile_id}.png")

                cv2.imwrite(tile_path, cv2.cvtColor(tile_rgb_8bit, cv2.COLOR_RGB2BGR))

                metadata["tiles"].append({
                    "id": tile_id,
                    "x": x,
                    "y": y,
                    "path": tile_path,
                    "nodata_fraction": nodata_fraction
                })

                tile_id += 1

    # Guardar metadata
    with open(os.path.join(out_dir, "tiles_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("✔ Tiling finished.")
