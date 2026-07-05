#!/bin/bash
set -e

echo "=== Installing PostgreSQL 16 + pgvector in WSL ==="

# Add PostgreSQL 16 repo
sudo apt-get update
sudo apt-get install -y wget gnupg2 lsb-release
echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update

# Install PostgreSQL 16 + dev headers (needed to compile pgvector)
sudo apt-get install -y postgresql-16 postgresql-server-dev-16 build-essential git

# Install pgvector from source
cd /tmp
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
cd ~

# Start PostgreSQL
sudo service postgresql start

# Create database and user (matching docker-compose config)
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres psql -c "CREATE DATABASE concept_genealogy;" 2>/dev/null || echo "DB already exists"
sudo -u postgres psql -d concept_genealogy -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Allow connections from Windows host (localhost)
PG_HBA=$(sudo -u postgres psql -t -c "SHOW hba_file;" | xargs)
echo "host all all 0.0.0.0/0 md5" | sudo tee -a "$PG_HBA"

PG_CONF=$(sudo -u postgres psql -t -c "SHOW config_file;" | xargs)
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "$PG_CONF"

sudo service postgresql restart

echo "=== Done! PostgreSQL 16 + pgvector running on port 5432 ==="
echo "Connection: postgresql://postgres:postgres@localhost:5432/concept_genealogy"
