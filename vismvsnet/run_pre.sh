
DATA_ROOT="/home/gj/Project/Relightable3DGaussian/datasets"
WIDTH=800
HEIGHT=800
NUM_SRC=5

python convert.py -s $DATA_ROOT
python colmap2mvsnet.py --dense_folder $DATA_ROOT --max_d 256
python test.py --data_root $DATA_ROOT --resize $WIDTH,$HEIGHT --crop $WIDTH,$HEIGHT --num_src $NUM_SRC
python filter.py --data $DATA_ROOT/vis_mvsnet --pair $DATA_ROOT/pair.txt --view $NUM_SRC --vthresh 2 --pthresh '.6,.6,.6' --out_dir $DATA_ROOT/filtered