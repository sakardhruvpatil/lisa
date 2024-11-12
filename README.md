# lisa


## Steps to setup mongodb

pip install pymongo

sudo apt-get install gnupg curl

curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc |    sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg    --dearmor

echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

sudo apt-get update

sudo apt-get install -y mongodb-org

sudo systemctl start mongod

sudo systemctl enable mongod

sudo systemctl status mongod

mongosh

sudo nano /etc/sysctl.conf

    vm.max_map_count=128000
    vm.swappiness=0

mongosh

    use admin

    db.createUser({
    user: "lisa001",
    pwd: "sakar123", // Replace with a strong password
    roles: [ { role: "root", db: "lisa_db" } ]
    })

    use lisa_db

