#!/usr/bin/env bash

hatch build
hatch dep show requirements >requirements.txt
flyctl deploy --verbose --push --now
