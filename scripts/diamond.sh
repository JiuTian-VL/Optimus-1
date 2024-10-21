#!/bin/bash


python -m optimus1.main server.port=9000 benchmark=diamond evaluate="[0,1,2,3,4,5,6]" env.times=30 & 

wait