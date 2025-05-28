sudo docker compose build --build-arg UID="`id -u`" dev
sudo docker compose run --rm dev
