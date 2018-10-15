#!/bin/bash
docker run -it cuda-tf bash

#!/bin/bash
if [ "$1" != "" ]; then
    docker run -it "$1" bash 
else
    docker run -it cuda-tf bash
fi