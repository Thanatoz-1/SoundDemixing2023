from mdx.dataloaders.audiodataloader import AudioDataset

data = AudioDataset(
    ["./data/train/"],
    sampling_rate=[44100],
    sources=["drums", "vocals"],
    targets=["drums", "vocals", "bass"],
    transforms=None,
    ext="wav",
    debug=True,
)
p = data._get_mixture(
    indexes=[
        ("./data/train/Actions - Devil's Words", "drum", 0.5),
        ("./data/train/Alexander Ross - Velvet Curtain", "vocals", 0.3),
        ("./data/train/Alexander Ross - Goodbye Bolero", "vocals", 0.22),
    ]
)
inp = p["input"]
out = p["output"]
# inp.min(), inp.max(), out["vocals"].min(), out["vocals"].max()
