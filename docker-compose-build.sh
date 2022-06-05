#!/bin/bash

#copy utils module into the single api folders
cp -r ./utils ./prod-api/
cp -r ./utils ./trainer-api/
cp -r ./utils ./data-api/

# copy model in the directory. TODO Pull from registry
cp -r ./src/models/production.h5 ./trainer-api/models/candidate.h5
cp -r ./src/models ./prod-api/

#copy requirements.txt into the single api folders (adapt to single api req?)
cp requirements.txt ./prod-api/
cp requirements.txt ./trainer-api/
cp requirements.txt ./data-api/

docker-compose build --parallel

# delete all copied files
rm -r ./data-api/utils/
rm -r ./trainer-api/utils/
rm -r ./prod-api/utils/

# delete models directory
rm -r ./trainer-api/models
rm -r ./prod-api/models

# delete copied files
rm -r ./data-api/requirements.txt
rm -r ./trainer-api/requirements.txt
rm -r ./prod-api/requirements.txt