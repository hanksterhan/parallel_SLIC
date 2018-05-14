#! /bin/bash

for i in {1..5}
do
  python SLIC.py -s -k 100  -p -i input/frog2048.jpg;
  python SLIC.py -s -k 100  -p -i input/frog1024.jpg;
  python SLIC.py -s -k 100  -p -i input/frog512.jpg;
  python SLIC.py -s -k 100  -p -i input/frog256.jpg;
  python SLIC.py -s -k 100  -p -i input/frog128.jpg;
  python SLIC.py -s -k 100     -i input/frog2048.jpg;
  python SLIC.py -s -k 100     -i input/frog1024.jpg;
  python SLIC.py -s -k 100     -i input/frog512.jpg;
  python SLIC.py -s -k 100     -i input/frog256.jpg;
  python SLIC.py -s -k 100     -i input/frog128.jpg;
  python SLIC.py -s -k 1000 -p -i input/frog2048.jpg;
  python SLIC.py -s -k 1000 -p -i input/frog1024.jpg;
  python SLIC.py -s -k 1000 -p -i input/frog512.jpg;
  python SLIC.py -s -k 1000 -p -i input/frog256.jpg;
  python SLIC.py -s -k 1000 -p -i input/frog128.jpg;
  python SLIC.py -s -k 1000    -i input/frog2048.jpg;
  python SLIC.py -s -k 1000    -i input/frog1024.jpg;
  python SLIC.py -s -k 1000    -i input/frog512.jpg;
  python SLIC.py -s -k 1000    -i input/frog256.jpg;
  python SLIC.py -s -k 1000    -i input/frog128.jpg;
done
