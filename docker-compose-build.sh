#!/bin/bash
# TODO loop for scale

#copy utils module into the single api folders
cp -r ./utils ./prod-api/
cp -r ./utils ./trainer-api/
cp -r ./utils ./data-api/
cp -r ./utils ./staticprod-api/

# copy model in the directory. TODO Pull from registry
cp -r ./src/gcp_config ./trainer-api/
cp -r ./src/gcp_config ./prod-api/
cp -r ./src/gcp_config ./staticprod-api/

#copy requirements.txt into the single api folders (adapt to single api req?)
cp requirements.txt ./prod-api/
cp requirements.txt ./trainer-api/
cp requirements.txt ./data-api/
cp requirements.txt ./staticprod-api/

docker compose build --parallel

# delete all copied files
rm -r ./data-api/utils/
rm -r ./trainer-api/utils/
rm -r ./prod-api/utils/
rm -r ./staticprod-api/utils/

# delete models directory
rm -r ./trainer-api/gcp_config
rm -r ./prod-api/gcp_config
rm -r ./staticprod-api/gcp_config

# delete copied files
rm -r ./data-api/requirements.txt
rm -r ./trainer-api/requirements.txt
rm -r ./prod-api/requirements.txt
rm -r ./staticprod-api/requirements.txt