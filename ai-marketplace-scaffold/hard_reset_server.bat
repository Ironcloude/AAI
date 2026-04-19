REM #!/bin/bash
REM Remove containers and delete volumes
docker compose down -v

REM Rebuild images
REM docker compose build
REM Start the containers
REM docker compose up
docker compose up --build

