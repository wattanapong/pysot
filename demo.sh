ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

python tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot /media/wattanapongsu/3T/models/pysot/siamrpn_r50_l234_dwxcorr.pth