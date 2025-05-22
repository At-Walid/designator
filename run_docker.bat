@echo off
docker load -i designate-gui.tar
docker run -it --rm designate-gui

