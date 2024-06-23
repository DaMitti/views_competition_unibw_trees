#!/bin/bash
#

#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [-dl]"
    echo "  -d          Run pipeline in docker mode."
    echo "  -l          Watch logs of freshly started docker container."
    echo "Subcommands:"
    echo "  predict     Run the prediction pipeline (default)"
    echo "  evaluate    Run evaluation pipeline"
    echo "  logs        Watch logs of running docker container."
    echo "  stop        Stop runnign docker container."
    echo " "
    echo "Example:"
    echo "  ./run_pipeline.sh -d predict"
    exit 1
}

# Parse command-line options
docker=0
logmode=0
while getopts ":dl" opt; do
  case ${opt} in
    d )
      docker=1
      echo "sdadas"
      ;;
    l )
      logmode=1
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      usage
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      usage
      ;;
  esac
done
shift $((OPTIND -1))

prediction_pipeline() {
    if [ $docker -eq 1 ]; then
        echo "[::] Running pipeline in Docker environment..."
        if [ -z "$(docker images -q ccew-tree 2> /dev/null)" ]; then
            # Build the image if it doesn't exist.
            echo "[::] CCEW image doesn't exist. Building image first..."
            docker build -t ccew-tree . 
        fi
        echo "[::] Running competition pipeline code in detached docker mode..."
        container_id=$(docker ps -q --filter "ancestor=ccew-tree" | head -n 1)
        if [ $container_id  ]; then
            echo "[::] Container already running..."
        else
            container_id=$(docker run -d -v .:/usr/src/app ccew-tree "(cd src && PYTHONPATH=../ python3.11 -u competition_pipeline.py)")
        fi
        if [ $logmode -eq 1 ]; then
            docker logs $container_id -f
            exit 0
        fi
    else
        echo "[::] Running pipeline in local Python environment..."
        (cd src && PYTHONPATH=../ python3.11 competition_pipeline.py)
    fi
}

evaluation_pipeline() {
    if [ $docker -eq 1 ]; then
        echo "[::] Running evaluation pipeline in Docker environment..."
        if [ -z "$(docker images -q ccew-tree 2> /dev/null)" ]; then
            # Build the image if it doesn't exist.
            echo "[::] CCEW image doesn't exist. Building image first..."
            docker build -t ccew-tree . 
        fi
        echo "[::] Running evaluation pipeline code in detached docker mode..."
        container_id=$(docker ps -q --filter "ancestor=ccew-tree" | head -n 1)
        if [ $container_id  ]; then
            echo "[::] Container already running..."
        else
            container_id=$(docker run -d -v .:/usr/src/app ccew-tree "(cd src && PYTHONPATH=../ python3.11 -u evaluation_pipeline.py)")
        fi
        if [ $logmode -eq 1 ]; then
            docker logs $container_id -f
            exit 0
        fi
    else
        echo "[::] Running evaluation pipeline in local Python environment..."
        (cd src && PYTHONPATH=../ python3.11 evaluation_pipeline.py)
    fi
}


subcommand=$1;

case "$subcommand" in
    "" )
        prediction_pipeline
        ;;
    predict )
        prediction_pipeline
        ;;
    evaluate )
        evaluation_pipeline
        ;;
    logs )
        container_id=$(docker ps -q --filter "ancestor=ccew-tree" | head -n 1)
        if [ $container_id  ]; then
            docker logs $container_id -f
            exit 0
        else
            echo "[::] No container is running. Start script in docker mode first!"
            exit 1
        fi
        ;;
    stop )
        container_id=$(docker ps -q --filter "ancestor=ccew-tree" | head -n 1)
        if [ $container_id  ]; then
            echo "[::] Stopping docker container..."
            docker stop $container_id
        fi
        ;;
    * ) 
        echo "Invalid subcommand: $subcommand -$OPTARG" 1>&2
        usage
        ;;
esac


