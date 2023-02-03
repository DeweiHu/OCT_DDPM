#!/usr/bin/env bash

Dir=/home/dewei/Desktop/fish/temp
jif=/home/dewei/tool/oguzi/bin/label_fusion
c3d=/home/dewei/src/c3d-1.0.0-Linux-x86_64/bin/c3d

target=$Dir/fix_img.tif
atlases=$Dir/atlas*.tif
weights=$Dir/weights%02d.tif
synthResult=$Dir/synthResult.tif

$jif 2 -g $atlases -m Joint[30,2] -rp 5x5 -rs 0x0 -w $weights  $target $synthResult # >> $Dir/temp.log
$c3d $Dir/weight* $atlases -reorder 0.5 -wsv -o $synthResult
