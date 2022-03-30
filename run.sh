echo "Se lanzará el proceso automatizado de ejecución de la aplicación"

echo "Crear red fayspy"
docker network create fayspy

echo "Añadiendo contenedor mysql_server a la red"
docker pull mysql:oracle
docker run --network fayspy --network-alias mysql_server --name mysql_server -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE=FaysPy -d mysql:oracle

echo "Añadiendo contenedor phpmyadmin a la red"
docker pull phpmyadmin/phpmyadmin:latest
docker run --network fayspy --network-alias phpmyadmin --name phpmyadmin -d --link mysql_server:db -p 8081:80 phpmyadmin/phpmyadmin

echo "Añadiendo contenedor portainer a la red"
docker run --network fayspy --network-alias portainer --name portainer -d -p 9000:9000 -v /var/run/docker.sock:/var/run/docker.sock portainer/portainer
echo "Esperar 1 minuto para levantar el contenedor de la aplicación..."
time(5)

echo "Construyendo el contenedor Docker de la aplicación."
#docker build -t sergiomasavi/fayspy-apps-ml-tf:latest . # https://luis-sena.medium.com/creating-the-perfect-python-dockerfile-51bdec41f1c8
docker pull sergiomasavi/fayspy-apps-ml-tf:latest
echo "Ejecutando contenedor de la aplicación"
docker run --network fayspy --network-alias fayspy --name fayspy --link mysql_server:db -h 0.0.0.0 -p 5000:5000 -d sergiomasavi/fayspy-apps-ml-tf:latest
echo "Contenedores ejecutados. Compruebe su funcionameinto."