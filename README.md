# mydocreaderCore
Template Validation and Text Extraction APIs

## Requirements
- docker
- docker-compose

## Get Started

- Clone the repository.
- Update the `production_deployment` (boolean), `fileserver` endpoint and `server_host` accordingly in `src/config.ini` file.
- Update `HOST` and `hostname` in `docker-compose.yaml` file.
- Go to the directory of the file docker-compose.yaml.


Run the following command to run the application - 

```bash
docker image prune && docker-compose build && docker-compose up 
```

To stop the application, run - 

```bash
docker-compose down 
```