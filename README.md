# NEED dam break simulation

## Dependencies

- fenics
- meshio

To install FEniCS, use one of the methods described here:
<https://fenicsproject.org/download/archive/>.
In case of using the Docker image, in addition to `meshio` it is required
to install `h5py`.
In case of using Anaconda, it might be necessary to install `mshr` library:
<https://github.com/martinal/mshr>.

## Running

To run the simulation, execute
```
python3 tsunami.py
```
Results produced by the code can be found in `tsunami.xdmf` file.
