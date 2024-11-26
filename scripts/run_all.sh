#!/bin/sh

### Synthetic
# Ellipse
python -m nfe.train --experiment synthetic --data ellipse --model flow --flow-model resnet --solver dopri5
python -m nfe.train --experiment synthetic --data ellipse --model flow --flow-model coupling --solver dopri5
python -m nfe.train --experiment synthetic --data ellipse --model flow --flow-model gru --solver dopri5
python -m nfe.train --experiment synthetic --data ellipse --model flow --flow-model mlp --solver dopri5