
import sys
import struct

_IOC_SIZEBITS = 14
_IOC_SIZEMASK = (1 << _IOC_SIZEBITS) - 1
_IOC_NRSHIFT = 0
_IOC_NRBITS = 8
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_TYPEBITS = 8
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
IOCSIZE_MASK = _IOC_SIZEMASK << _IOC_SIZESHIFT
IOCSIZE_SHIFT = _IOC_SIZESHIFT

# Python 2.2 uses a signed int for the ioctl() call, so ...
if (sys.version_info[0] < 3) or (sys.version_info[1] < 3):
    _IOC_WRITE = 1
    _IOC_READ = -2
    _IOC_INOUT = -1
else:
    _IOC_WRITE = 1
    _IOC_READ = 2
    _IOC_INOUT = 3

_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS


def sizeof(type):
    return struct.calcsize(type)


def _IOC(dir, type, nr, size):
    return int((dir << _IOC_DIRSHIFT) | (type << _IOC_TYPESHIFT) |
               (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT))


def _IO(type, nr):
    return _IOC(_IOC_NONE,  type, nr, 0)


def _IOR(type, nr, size):
    return _IOC(_IOC_READ,  type, nr, sizeof(size))


def _IOW(type, nr, size):
    return _IOC(_IOC_WRITE, type, nr, sizeof(size))


def _IOWR(type, nr, size):
    return _IOC(_IOC_READ | _IOC_WRITE, type, nr, sizeof(size))


NCS_MAGIC = ord('n')

RECACCEL_IOCTL_ACQUIRE_BUFFER = 1
RECACCEL_IOCTL_PREDICT = 2
RECACCEL_IOCTL_RELEASE_BUFFER = 3

NCS_IOCTL_ARG = 'IIQQ'
RECACCEL_IOCX_ACQUIRE_BUFFER =\
    _IOWR(NCS_MAGIC, RECACCEL_IOCTL_ACQUIRE_BUFFER, NCS_IOCTL_ARG)
RECACCEL_IOCX_PREDICT =\
    _IOWR(NCS_MAGIC, RECACCEL_IOCTL_PREDICT, NCS_IOCTL_ARG)
RECACCEL_IOCX_RELEASE_BUFFER =\
    _IOW(NCS_MAGIC, RECACCEL_IOCTL_RELEASE_BUFFER, NCS_IOCTL_ARG)

NFUNC_DLRM = 0
