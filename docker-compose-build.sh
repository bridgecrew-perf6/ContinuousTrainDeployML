#!/bin/bash

#copy utils module into the single api folders
cp -r ./utils ./prod-api/
cp -r ./utils ./trainer/
cp -r ./utils ./data-api/

# copy model in the directory. TODO Pull from registry
cp -r ./src/models ./trainer/
cp -r ./src/models ./prod-api/

#copy requirements.txt into the single api folders (adapt to single api req?)
cp requirements.txt ./prod-api/
cp requirements.txt ./trainer/
cp requirements.txt ./data-api/

docker-compose build --parallel

# delete all copied files
rm -r ./data-api/utils/
rm -r ./trainer/utils/
rm -r ./prod-api/utils/

# delete models directory
rm -r ./trainer/models
rm -r ./prod-api/models

# delete copied files
rm -r ./data-api/requirements.txt
rm -r ./trainer/requirements.txt
rm -r ./prod-api/requirements.txt