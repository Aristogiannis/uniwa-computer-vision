from cv_diffusion.preprocessing.dataset import (
    DisasterImageDataset,
    SatelliteFolderDataset,
    build_text_prompt,
)


def test_satellite_folder_dataset(synthetic_dataset_dir):
    ds = SatelliteFolderDataset(synthetic_dataset_dir)
    assert len(ds) == 8
    assert ds.classes == ["flood", "wildfire"]
    image, label = ds[0]
    assert image.mode == "RGB"
    assert label in (0, 1)


def test_disaster_image_dataset(synthetic_dataset_dir):
    ds = DisasterImageDataset(root=synthetic_dataset_dir)
    assert len(ds) == 8
    item = ds[0]
    assert {"pixel_values", "caption", "category"} <= set(item)
    assert isinstance(item["caption"], str) and len(item["caption"]) > 0


def test_prompt_template_deterministic():
    assert build_text_prompt("flood", index=0) == build_text_prompt("flood", index=0)
    assert "satellite" in build_text_prompt("flood")
    assert "wildfire" in build_text_prompt("wildfire")
