# pylint: disable=invalid-name, unused-import
"""Runtime NDArray api"""
from __future__ import absolute_import

import sys
import ctypes
import numpy as np
from .base import _LIB, check_call, c_array, string_types, _FFI_MODE, c_str
from .runtime_ctypes import DECORDType, DECORDContext, DECORDArray, DECORDArrayHandle
from .runtime_ctypes import TypeCode, decord_shape_index_t


IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError

try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    if sys.version_info >= (3, 0):
        from ._cy3.core import _set_class_ndarray, _reg_extension, _make_array, _from_dlpack
        from ._cy3.core import NDArrayBase as _NDArrayBase
    else:
        from ._cy2.core import _set_class_ndarray, _reg_extension, _make_array, _from_dlpack
        from ._cy2.core import NDArrayBase as _NDArrayBase
except IMPORT_EXCEPT:
    # pylint: disable=wrong-import-position
    from ._ctypes.ndarray import _set_class_ndarray, _reg_extension, _make_array, _from_dlpack
    from ._ctypes.ndarray import NDArrayBase as _NDArrayBase

def context(dev_type, dev_id=0):
    """Construct a DECORD context with given device type and id.

    Parameters
    ----------
    dev_type: int or str
        The device type mask or name of the device.

    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx: DECORDContext
        The corresponding context.

    Examples
    --------
    Context can be used to create reflection of context by
    string representation of the device type.

    .. code-block:: python

      assert decord.context("cpu", 1) == decord.cpu(1)
      assert decord.context("gpu", 0) == decord.gpu(0)
      assert decord.context("cuda", 0) == decord.gpu(0)
    """
    if isinstance(dev_type, string_types):
        dev_type = dev_type.split()[0]
        if dev_type not in DECORDContext.STR2MASK:
            raise ValueError("Unknown device type %s" % dev_type)
        dev_type = DECORDContext.STR2MASK[dev_type]
    return DECORDContext(dev_type, dev_id)


def numpyasarray(np_data):
    """Return a DECORDArray representation of a numpy array.
    """
    data = np_data
    assert data.flags['C_CONTIGUOUS']
    arr = DECORDArray()
    shape = c_array(decord_shape_index_t, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.strides = None
    arr.dtype = DECORDType(np.dtype(data.dtype).name)
    arr.ndim = data.ndim
    # CPU device
    arr.ctx = context(1, 0)
    return arr, shape


def empty(shape, dtype="float32", ctx=context(1, 0)):
    """Create an empty array given shape and device

    Parameters
    ----------
    shape : tuple of int
        The shape of the array

    dtype : type or str
        The data type of the array.

    ctx : DECORDContext
        The context of the array

    Returns
    -------
    arr : decord.nd.NDArray
        The array decord supported.
    """
    shape = c_array(decord_shape_index_t, shape)
    ndim = ctypes.c_int(len(shape))
    handle = DECORDArrayHandle()
    dtype = DECORDType(dtype)
    check_call(_LIB.DECORDArrayAlloc(
        shape, ndim,
        ctypes.c_int(dtype.type_code),
        ctypes.c_int(dtype.bits),
        ctypes.c_int(dtype.lanes),
        ctx.device_type,
        ctx.device_id,
        ctypes.byref(handle)))
    return _make_array(handle, False)


def from_dlpack(dltensor):
    """Produce an array from a DLPack tensor without memory copy.
    Retrieves the underlying DLPack tensor's pointer to create an array from the
    data. Removes the original DLPack tensor's destructor as now the array is
    responsible for destruction.

    Parameters
    ----------
    dltensor : DLPack tensor
        Input DLManagedTensor, can only be consumed once.

    Returns
    -------
    arr: decord.nd.NDArray
        The array view of the tensor data.
    """
    return _from_dlpack(dltensor)


# ═══════════════ DLPack export support (see NDArrayBase.__dlpack__) ═══════════════

class _DLPackHolder:
    """Owns everything backing an exported DLManagedTensor.

    The holder is registered in _DLPACK_HOLDERS under the address of its
    DLManagedTensor; the deleter callback (invoked by numpy once the
    produced array is garbage collected) removes it, dropping the reference
    to the source NDArray and the DLManagedTensor memory.
    """
    __slots__ = ("ndarray", "dlm")


class _DLManagedTensor(ctypes.Structure):
    """DLPack DLManagedTensor (DLTensor + manager_ctx + deleter)."""
    _fields_ = [
        ("dl_tensor", DECORDArray),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_DLPACK_HOLDERS: dict = {}


def _dlpack_deleter(addr):
    holder = _DLPACK_HOLDERS.pop(addr, None)
    if holder is None:
        return
    holder.ndarray = None
    holder.dlm = None


_DLPACK_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(_dlpack_deleter)
# 64-bit pointer safety for the Python C API call below; restype py_object
# so ctypes adopts the new reference returned by PyCapsule_New.
ctypes.pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


class NDArrayBase(_NDArrayBase):
    """A simple Device/CPU Array object in runtime."""
    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i] for i in range(self.handle.contents.ndim))

    @property
    def dtype(self):
        """Type of this array"""
        return str(self.handle.contents.dtype)

    @property
    def ctx(self):
        """context of this array"""
        return self.handle.contents.ctx

    @property
    def context(self):
        """context of this array"""
        return self.ctx

    def __hash__(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Check object identity equality

        Parameters
        ----------
        other : object
            The other object to compare to

        Returns
        -------
        same : bool
            Whether other is same as self.
        """
        if not isinstance(other, NDArrayBase):
            return False
        return self.__hash__() == other.__hash__()

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArrayBase):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self.copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def copyfrom(self, source_array):
        """Perform a synchronized copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.

        Returns
        -------
        arr : NDArray
            Reference to self.
        """
        if isinstance(source_array, NDArrayBase):
            source_array.copyto(self)
            return self

        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(source_array)))
        t = DECORDType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError("array shape do not match the shape of NDArray {0} vs {1}".format(
                source_array.shape, shape))
        source_array = np.ascontiguousarray(source_array, dtype=dtype)
        assert source_array.flags['C_CONTIGUOUS']
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(source_array.size * source_array.dtype.itemsize)
        check_call(_LIB.DECORDArrayCopyFromBytes(self.handle, data, nbytes))
        return self

    def __repr__(self):
        res = "<decord.NDArray shape={0}, {1}>\n".format(self.shape, self.context)
        res += self.asnumpy().__repr__()
        return res

    def __str__(self):
        return str(self.asnumpy())

    def asnumpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        t = DECORDType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags['C_CONTIGUOUS']
        data = np_arr.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
        check_call(_LIB.DECORDArrayCopyToBytes(self.handle, data, nbytes))
        return np_arr

    # ═══════════════ DLPack zero-copy export (numpy 2.x protocol) ═══════════════
    # np.from_dlpack(arr) → zero-copy view of the decord buffer.  The view
    # references this NDArray (kept alive by _DLPACK_HOLDERS); when numpy
    # releases it, the deleter drops the reference and the buffer returns to
    # its pool / is freed.  CPU arrays only — GPU arrays must go through
    # next_roi (D2H ROI copy), numpy has no CUDA support in from_dlpack.
    def __dlpack_device__(self):
        """DLPack device tuple (device_type, device_id), numpy protocol."""
        ctx = self.ctx
        return (int(ctx.device_type), int(ctx.device_id))

    def __dlpack__(self, stream=None):
        """Export as a DLManagedTensor capsule (numpy 2.x protocol).

        Returns a PyCapsule named "dltensor" wrapping a DLManagedTensor that
        aliases this array's buffer.  The capsule (and the produced numpy
        view) keep the source NDArray alive until released.
        """
        if stream not in (None, 0):
            raise ValueError("decord NDArray supports only stream=None")
        ctx = self.ctx
        if int(ctx.device_type) != 1:  # kDLCpu
            raise RuntimeError(
                "__dlpack__ supports CPU arrays only (got device type {}); "
                "use next_roi() for GPU frames".format(ctx.device_type))
        if self.handle is None or not self.handle.contents.data:
            raise RuntimeError("cannot export an empty NDArray via __dlpack__")
        t = DECORDType(self.dtype)
        if t.lanes > 1:
            raise RuntimeError("__dlpack__ does not support vector lanes")

        # DLManagedTensor: DLTensor + manager_ctx + deleter.  The DLTensor is
        # copied shallowly (pointers alias this array's buffers, which stay
        # alive through the holder's reference to self).
        dlm = _DLManagedTensor()
        dlm.dl_tensor = self.handle.contents
        dlm.manager_ctx = ctypes.c_void_p(0)
        addr = ctypes.addressof(dlm)
        holder = _DLPackHolder()
        holder.ndarray = self
        holder.dlm = dlm
        _DLPACK_HOLDERS[addr] = holder
        dlm.deleter = ctypes.cast(_DLPACK_DELETER, ctypes.c_void_p)
        # Capsule destructor must be NULL: per the DLPack protocol the
        # consumer (numpy) owns the release and calls dlmt->deleter when
        # the produced array is GC'd.  A capsule destructor here would free
        # the buffer while numpy still aliases it.
        capsule = ctypes.pythonapi.PyCapsule_New(
            ctypes.c_void_p(addr), ctypes.c_char_p(b"dltensor"),
            ctypes.c_void_p())
        if not capsule:
            _DLPACK_HOLDERS.pop(addr, None)
            raise RuntimeError("PyCapsule_New failed")
        return capsule

    def copyto(self, target):
        """Copy array to target

        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.
        """
        if isinstance(target, DECORDContext):
            target = empty(self.shape, self.dtype, target)
        if isinstance(target, NDArrayBase):
            check_call(_LIB.DECORDArrayCopyFromTo(
                self.handle, target.handle, None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target


def free_extension_handle(handle, type_code):
    """Free c++ extension type handle

    Parameters
    ----------
    handle : ctypes.c_void_p
        The handle to the extension type.

    type_code : int
         The tyoe code
    """
    check_call(_LIB.DECORDExtTypeFree(handle, ctypes.c_int(type_code)))

def register_extension(cls, fcreate=None):
    """Register a extension class to DECORD.

    After the class is registered, the class will be able
    to directly pass as Function argument generated by DECORD.

    Parameters
    ----------
    cls : class
        The class object to be registered as extension.

    Note
    ----
    The registered class is requires one property: _decord_handle and a class attribute _decord_tcode.

    - ```_decord_handle``` returns integer represents the address of the handle.
    - ```_decord_tcode``` gives integer represents type code of the class.

    Returns
    -------
    cls : class
        The class being registered.

    fcreate : function, optional
        The creation function to create a class object given handle value.

    Example
    -------
    The following code registers user defined class
    MyTensor to be DLTensor compatible.

    .. code-block:: python

       @decord.register_extension
       class MyTensor(object):
           _decord_tcode = decord.TypeCode.ARRAY_HANDLE

           def __init__(self):
               self.handle = _LIB.NewDLTensor()

           @property
           def _decord_handle(self):
               return self.handle.value
    """
    if fcreate and cls._decord_tcode < TypeCode.EXT_BEGIN:
        raise ValueError("Cannot register create when extension tcode is same as buildin")
    _reg_extension(cls, fcreate)
    return cls
