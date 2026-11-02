| Dataset | data mean | data max | BC | %BC (top 10%) | DT, target = 1.0 x data max | DT, best target (oracle) | seeds |
|---|---|---|---|---|---|---|---|
| hopper-medium | 87.2 | 120.7 | 103.7 +- 4.4 | 104.5 +- 6.9 | 70.5 +- 20.5 | 70.8 +- 16.9 (x0.8) | 3 |
| halfcheetah-medium | 99.6 | 116.9 | 54.0 +- 19.7 | 118.4 +- 6.5 | 33.5 +- 16.3 | 34.4 +- 1.2 (x1.2) | 3 |
| walker2d-medium | 124.2 | 135.0 | 134.7 +- 0.1 | 135.7 +- 0.6 | 131.3 +- 1.8 | 131.5 +- 1.1 (x0.6) | 3 |
| hopper-simple | 51.5 | 102.0 | 98.3 +- 0.3 | 98.7 +- 0.2 | 80.1 +- 32.1 | 87.8 +- 13.9 (x0.4) | 3 |
| hopper-medium-expert | 101.6 | 135.1 | 113.1 +- 0.9 | 105.7 +- 8.5 | 56.8 +- 15.1 | 68.3 +- 18.0 (x0.4) | 3 |

Per-seed means (normalized):
hopper-medium bc: 107.6, 98.9, 104.5
hopper-medium pbc: 107.0, 109.8, 96.7
hopper-medium dt: 78.4, 47.3, 85.9
halfcheetah-medium bc: 76.4, 46.1, 39.5
halfcheetah-medium pbc: 125.0, 118.2, 112.0
halfcheetah-medium dt: 47.2, 15.4, 37.9
walker2d-medium bc: 134.6, 134.6, 134.8
walker2d-medium pbc: 135.1, 136.0, 136.2
walker2d-medium dt: 131.9, 129.3, 132.8
hopper-simple bc: 98.5, 98.0, 98.2
hopper-simple pbc: 98.7, 98.8, 98.5
hopper-simple dt: 43.0, 98.5, 98.9
hopper-medium-expert bc: 112.5, 112.6, 114.2
hopper-medium-expert pbc: 103.6, 115.1, 98.4
hopper-medium-expert dt: 57.5, 41.3, 71.4
