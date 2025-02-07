ssh <votre_nom_utilisateur>@<4.211.204.4>
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
docker pull votre_nom_utilisateur/train_model:latest
docker run votre_nom_utilisateur/train_model:latest

