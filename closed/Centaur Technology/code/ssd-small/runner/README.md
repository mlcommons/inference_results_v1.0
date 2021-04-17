# MLPerf Runner

This is a simple wrapper around the MLPerf loadgen module to be able to run
and submit MLPerf results using our own custom CC runner.

Currently, MLPerf only provides backends that utilize the Python API's but we
lose some performance due to the "two language" problem of converting to /
from Python and C++. In addition, better control over threading instead of
relying on tensorflow / MLPerf model harness' will allow us to squeak out
some more performance from models that we already support.