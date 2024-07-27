# teaser
python relighting.py -co configs/teaser --output "output/relighting/teaser_trace" -e "env_map/teaser.hdr" --sample 384
python relighting.py -co configs/teaser --output "output/relighting/teaser_refine" -e "env_map/teaser.hdr" --sample 24 --bake
# for nerf_syn dataset
python relighting.py -co configs/nerf_syn --video --output "output/relighting/nerf_syn" -e "env_map/composition.hdr" --sample_num 384
python relighting.py -co configs/nerf_syn_light --video --output "output/relighting/nerf_syn_light" -e "env_map/composition.hdr" --sample_num 384
# for tanks and temples dataset
python relighting.py -co configs/tnt --video --output "output/relighting/tnt" -e "env_map/ocean_from_horn.jpg" --sample_num 384
python relighting.py -co configs/tnt --video --output "output/relighting/tnt_shadow" -e "env_map/envmap_object_composition.hdr" --sample_num 384