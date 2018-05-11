#! /bin/bash

python SLIC.py -p -i boundary_recall/original_images/00000001.jpg -b -f boundary_recall/boundaries/00000001p.png -k 100 &
python SLIC.py -p -i boundary_recall/original_images/00000002.jpg -b -f boundary_recall/boundaries/00000002p.png -k 100 &
python SLIC.py -p -i boundary_recall/original_images/00000004.jpg -b -f boundary_recall/boundaries/00000004p.png -k 100 &
python SLIC.py -p -i boundary_recall/original_images/00000005.jpg -b -f boundary_recall/boundaries/00000005p.png -k 100 &
python SLIC.py -p -i boundary_recall/original_images/00000009.jpg -b -f boundary_recall/boundaries/00000009p.png -k 100 &
