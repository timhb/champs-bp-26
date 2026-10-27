# Dependencies

Every direct dependency and why we need it. Versions live in `pixi.lock`.

| Package | Purpose |
| --- | --- |
| python | Pinned to 3.14 so everyone runs the same interpreter |
| pixi | Environments, lockfile and tasks in one tool; changes are reviewable in PRs |
| ruff | Linting, import sorting, formatting |
| openslide-python | SVS viewer: reads `.svs` slides (tiles, thumbnails, label/macro images, metadata); brings the OpenSlide C library |
| flask | SVS viewer: local web server for pages, tiles and JSON endpoints |
| pillow | SVS viewer: encodes tiles, thumbnails and snapshots |
| pandas | SVS viewer: loads the case CSVs, CPL reports and derived tables |
| pyarrow | Parquet engine for pandas (`slide_labels.parquet`, `pigment_confirmation.parquet`) |
| numpy | Inference: feature arrays and pooled embeddings |
| scikit-learn | Attention graph: PCA and t-SNE projections of slide embeddings |
| umap-learn | Attention graph: UMAP projections of slide embeddings |
| python-igraph | Attention graph: Leiden communities on the full cosine neighbor graph |
| h5py | Inference: reads Virchow2 patch features and writes CLAM embeddings |
| pytorch | Inference: loads the stored CLAM-MB checkpoints and scores feature bags |
| torchvision | Inference: matching PyTorch vision operators used by Trident |
| py-opencv | Inference: native OpenCV libraries for Trident on the cluster |
| trident | Inference: tissue segmentation, tiling and Virchow2 feature extraction (source pipeline revision) |
| pypdf | SVS viewer: reads TAC assay names from the dataset description PDF |

The viewer pages also load OpenSeadragon 4.1.1, d3 7.9.0, topojson-client 3.1.0 and world-atlas 2.0.2 from pinned CDN URLs (not Pixi dependencies).
