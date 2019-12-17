
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <algorithm>

// Forward function declaration.
static PyObject *shuffle(PyObject *self, PyObject *args);
static PyObject *create_batch(PyObject *self, PyObject *args);
static PyObject *get_X(PyObject *self, PyObject *args);
static PyObject *get_y_lower_in(PyObject *self, PyObject *args);
static PyObject *get_y_upper_in(PyObject *self, PyObject *args);
static PyObject *get_y_lower_out(PyObject *self, PyObject *args);
static PyObject *get_y_upper_out(PyObject *self, PyObject *args);

// Boilerplate: method list.
static PyMethodDef methods[] = {
    { "shuffle", shuffle, METH_VARARGS, "Doc string."},
    { "create_batch", create_batch, METH_NOARGS, "Doc string."},
    { "get_X", get_X, METH_NOARGS, "Doc string."},
    { "get_y_lower_in", get_y_lower_in, METH_NOARGS, "Doc string."},
    { "get_y_upper_in", get_y_upper_in, METH_NOARGS, "Doc string."},
    { "get_y_lower_out", get_y_lower_out, METH_NOARGS, "Doc string."},
    { "get_y_upper_out", get_y_upper_out, METH_NOARGS, "Doc string."},
    { NULL, NULL, 0, NULL } /* Sentinel */
};

typedef npy_float32 pixel_type;
typedef npy_float32 label_type;
#define FRAME_WIDTH 25
#define FRAME_HEIGHT 20
#define FRAME_PIXEL_COUNT (FRAME_WIDTH*FRAME_HEIGHT)

// Boilerplate: Module initialization.
// Compare http://python3porting.com/cextensions.html
#if PY_MAJOR_VERSION >= 3
    #define MOD_ERROR_VAL NULL
    #define MOD_SUCCESS_VAL(val) val
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
                    static struct PyModuleDef moduledef = { \
                        PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
                    ob = PyModule_Create(&moduledef);
#else
    #define MOD_ERROR_VAL
    #define MOD_SUCCESS_VAL(val)
    #define MOD_INIT(name) void init##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
                    ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(sequence_shuffle)
{
        // deactivate this for debugging
        srand (time(NULL));

        PyObject *m;

        MOD_DEF(m, "sequence_shuffle", "some doc here", methods)

        if (m == NULL)
            return MOD_ERROR_VAL;

        import_array();

        return MOD_SUCCESS_VAL(m);

}

static size_t* make_shuffled_ids (size_t n) {
    if(n > RAND_MAX){
        printf("n > RAND_MAX\n");
        exit(1);
    }

    size_t* retval = (size_t*)malloc(sizeof(size_t)*n);

    for(size_t i=0; i<n; i++){
        retval[i] = i;
    }

    // TODO rework this
    for (size_t i = 0; i < n; i++) {
        size_t j = rand() % n;
        size_t tmp = retval[j];
        retval[j] = retval[i];
        retval[i] = tmp;
    }

    return retval;
}

#define data(x0) (*(pixel_type*)(((pixel_type*)PyArray_DATA(py_data) + \
                                (x0) * PyArray_STRIDES(py_data)[0])))

#define offsets(x0) (*(npy_uint64*)(( ((uint8_t*)PyArray_DATA(py_offsets)) + \
                                (x0) * PyArray_STRIDES(py_offsets)[0])))

#define labels_in(x0) (*(npy_uint64*)(( ((uint8_t*)PyArray_DATA(py_labels_in)) + \
                                (x0) * PyArray_STRIDES(py_labels_in)[0])))

#define labels_out(x0) (*(npy_uint64*)(( ((uint8_t*)PyArray_DATA(py_labels_out)) + \
                                (x0) * PyArray_STRIDES(py_labels_out)[0])))

static struct data_struct {
    size_t stream_count;
    size_t* stream_pixel_counts;
    size_t* stream_pixel_offsets;
    pixel_type* data;
    npy_float32* labels_in;
    npy_float32* labels_out;
} data;

static struct batch_struct {
    size_t current_stream;
    size_t batch_size;
    size_t batch_count;
    size_t batch_id;
    size_t frame_count;
    
    size_t* aggregation_lengths;
    size_t* shuffled_ids;

    pixel_type* X;
    npy_float32* y_lower_in;
    npy_float32* y_upper_in;
    npy_float32* y_lower_out;
    npy_float32* y_upper_out;
} batch;

/*****************************************************************************
 * shuffle                                                                   *
 *****************************************************************************/
static PyObject *shuffle(PyObject *self, PyObject *args) {

    memset(&batch, 0, sizeof(struct batch_struct));
    memset(&data, 0, sizeof(struct data_struct));

    npy_int64 aggregation_length_min;
    npy_int64 aggregation_length_max;
    PyArrayObject *py_data;
    PyArrayObject *py_offsets;
    PyArrayObject *py_labels_in;
    PyArrayObject *py_labels_out;

    if (!PyArg_ParseTuple(
        args, "lllO!O!O!O!",
        &batch.batch_size,
        &aggregation_length_min,
        &aggregation_length_max,
        &PyArray_Type, &py_data,
        &PyArray_Type, &py_offsets,
        &PyArray_Type, &py_labels_in,
        &PyArray_Type, &py_labels_out
    )) {
        return NULL;
    }

    batch.batch_id = 0;

    data.stream_count = PyArray_SIZE(py_offsets);
    size_t data_length = PyArray_SIZE(py_data);
    data.data = (pixel_type*)PyArray_DATA(py_data);

    size_t aggregation_length_delta = aggregation_length_max - aggregation_length_min;
    size_t remaining_count = data.stream_count;

    batch.aggregation_lengths = (size_t*)realloc(
        batch.aggregation_lengths,
        sizeof(size_t)*(data.stream_count/aggregation_length_min+1)
    );
    size_t aggregation_count = 0;

    while(remaining_count > 0){
        size_t tmp = (size_t)fmin(
            aggregation_length_min + (rand() % (aggregation_length_delta + 1)),
            remaining_count
        );
        remaining_count -= tmp;
        batch.aggregation_lengths[aggregation_count] = tmp;
        aggregation_count++;
    }

    data.stream_pixel_counts = (size_t*)realloc(
        data.stream_pixel_counts, sizeof(size_t)*data.stream_count
    );

    data.stream_pixel_offsets = (size_t*)realloc(
        data.stream_pixel_offsets, sizeof(size_t)*data.stream_count
    );

    data.labels_in = (npy_float32*)realloc(
        data.labels_in, sizeof(npy_float32)*data.stream_count
    );

    data.labels_out = (npy_float32*)realloc(
        data.labels_out, sizeof(npy_float32)*data.stream_count
    );

    for(size_t i=0; i<data.stream_count; i++){
        data.stream_pixel_counts[i] = i == data.stream_count - 1 ?
            data_length - offsets(i) : offsets(i+1)-offsets(i);
        data.stream_pixel_offsets[i] = offsets(i);
        data.labels_in[i] = labels_in(i);
        data.labels_out[i] = labels_out(i);
    }

    free(batch.shuffled_ids);
    batch.shuffled_ids = make_shuffled_ids(data.stream_count);
    batch.batch_count = aggregation_count / batch.batch_size;
    batch.current_stream = 0;

    Py_RETURN_NONE;

}

static PyObject *create_batch(PyObject *self, PyObject *args) {

    if(batch.batch_id == batch.batch_count){
        Py_RETURN_NONE;
    }

    auto old_current_stream = batch.current_stream;
    size_t longest_merged_stream_pixel_count = 0;

    for(size_t j = 0; j < batch.batch_size; j++){
        size_t aggregation_length = batch.aggregation_lengths[j];
        size_t stream_pixel_length_sum = 0;
        for(size_t k = 0; k < aggregation_length; k++){
            size_t shuffled_id = batch.shuffled_ids[
                batch.current_stream + k
            ];
            stream_pixel_length_sum += data.stream_pixel_counts[shuffled_id];
        }
        longest_merged_stream_pixel_count = (size_t)fmax(
            longest_merged_stream_pixel_count,
            stream_pixel_length_sum
        );
        batch.current_stream += aggregation_length;
    }

    batch.X = (pixel_type*)realloc(
        batch.X, sizeof(pixel_type)*
        batch.batch_size*longest_merged_stream_pixel_count
    );

    auto longest_merged_stream_frame_count =
        longest_merged_stream_pixel_count/FRAME_PIXEL_COUNT;

    auto y_byte_count = sizeof(pixel_type)*batch.batch_size*
        longest_merged_stream_frame_count;

    auto y_ptrs = {
        &batch.y_lower_in, &batch.y_lower_out,
        &batch.y_upper_in, &batch.y_upper_out
    };

    for(auto ptr : y_ptrs){
        *ptr = (label_type*)realloc(*ptr, y_byte_count);
    }

    batch.current_stream = old_current_stream;

    for(size_t j = 0; j < batch.batch_size; j++){
        auto aggregation_length = batch.aggregation_lengths[j];
        size_t stream_pixel_sum = 0;
        size_t stream_frame_sum = 0;

        size_t aggregation_lower_bound_in = 0;
        size_t aggregation_upper_bound_in = 0;
        size_t aggregation_lower_bound_out = 0;
        size_t aggregation_upper_bound_out = 0;
        
        for(size_t k = 0; k < aggregation_length; k++){
            auto shuffled_id = batch.shuffled_ids[
                batch.current_stream + k
            ];
            auto label_in = data.labels_in[shuffled_id];
            auto label_out = data.labels_out[shuffled_id];

            // printf("%f %f\n", label_in, label_out);

            auto stream_pixel_count = data.stream_pixel_counts[shuffled_id];
            auto stream_frame_count = stream_pixel_count / FRAME_PIXEL_COUNT;
            auto data_pixel_offset = data.stream_pixel_offsets[shuffled_id];

            int mode = rand() % 4;
            size_t do_reverse = mode == 2 || mode == 3;

            aggregation_upper_bound_in += do_reverse ? label_out : label_in;
            aggregation_upper_bound_out += do_reverse ? label_in : label_out;

            for(auto tmp : y_ptrs){
                auto ptr = *tmp + stream_frame_sum + j*longest_merged_stream_frame_count;
                std::fill(ptr, ptr + stream_frame_count,
                    ptr == batch.y_lower_in ? aggregation_lower_bound_in :
                    ptr == batch.y_lower_out ? aggregation_lower_bound_out :
                    ptr == batch.y_upper_in ? aggregation_upper_bound_in :
                    aggregation_upper_bound_out
                );
            }

            switch(mode){
                // plain copy
                case 0:
                    memcpy(
                        &batch.X[stream_pixel_sum],
                        &data.data[data_pixel_offset],

                        stream_pixel_count*sizeof(pixel_type)
                    );
                    break;
                // mirror
                case 1:
                    for(size_t l = 0; l < stream_frame_count; l++){
                        auto frame_offset = l * FRAME_PIXEL_COUNT;
                        for(size_t m = 0; m < FRAME_WIDTH; m++){
                            for(size_t n = 0; n < FRAME_HEIGHT; n++){
                                batch.X[
                                    stream_pixel_sum + frame_offset +
                                    n * FRAME_WIDTH + m
                                ] = data.data[
                                    data_pixel_offset + frame_offset +
                                    n * FRAME_WIDTH + (FRAME_WIDTH - 1 - m)
                                ];
                            }
                        }
                    }
                    break;
                // reverse
                case 2:
                    for(size_t l = 0; l < stream_frame_count; l++){
                        memcpy(
                            &batch.X[
                                stream_pixel_sum + l * FRAME_PIXEL_COUNT
                            ],
                            &data.data[
                                data_pixel_offset
                                + (stream_frame_count - 1 - l) * FRAME_PIXEL_COUNT
                            ],

                            FRAME_PIXEL_COUNT*sizeof(pixel_type)
                        );
                    }
                    break;
                // reverse and mirror
                case 3:
                    for(size_t l = 0; l < stream_frame_count; l++){
                        for(size_t m = 0; m < FRAME_WIDTH; m++){
                            for(size_t n = 0; n < FRAME_HEIGHT; n++){
                                batch.X[
                                    stream_pixel_sum + (l * FRAME_PIXEL_COUNT) +
                                    n * FRAME_WIDTH + m
                                ] = data.data[
                                    data_pixel_offset + (stream_frame_count-1-l)
                                    * FRAME_PIXEL_COUNT +
                                    n * FRAME_WIDTH + (FRAME_WIDTH - 1 - m)
                                ];
                            }
                        }
                    }
                    break;
            }
            aggregation_lower_bound_in += label_in;
            aggregation_lower_bound_out += label_out;

            stream_pixel_sum += stream_pixel_count;
            stream_frame_sum += stream_frame_count;
        }

        size_t batch_X_ptr = j*longest_merged_stream_pixel_count;

        std::fill(
            batch.X + batch_X_ptr + stream_pixel_sum,
            batch.X + batch_X_ptr + longest_merged_stream_pixel_count,
            -2
        );

        size_t batch_frame_total = stream_pixel_sum / FRAME_PIXEL_COUNT;
        size_t batch_y_ptr = j*longest_merged_stream_frame_count;

        for(auto ptr : y_ptrs){
            std::fill(
                *ptr + batch_y_ptr + batch_frame_total,
                *ptr + batch_y_ptr + longest_merged_stream_frame_count,
                -2
            );
        }

        batch.current_stream += aggregation_length;
    }

    batch.frame_count = batch.batch_size*longest_merged_stream_pixel_count/FRAME_PIXEL_COUNT;
    batch.batch_id++;

    Py_RETURN_NONE;
}

static PyObject *get_X(PyObject *self, PyObject *args) {
    npy_intp dims[] = {(npy_int)batch.frame_count, (npy_int)FRAME_PIXEL_COUNT};

    return PyArray_Return((PyArrayObject *)PyArray_SimpleNewFromData(
        2, dims, NPY_FLOAT32, batch.X
    ));
}

static PyObject *get_Y (npy_float32 *ptr) {
    npy_intp dims[] = {(npy_int)batch.frame_count};

    return PyArray_Return((PyArrayObject *)PyArray_SimpleNewFromData(
        1, dims, NPY_FLOAT32, ptr
    ));
}

static PyObject *get_y_lower_in(PyObject *self, PyObject *args) {
    return get_Y(batch.y_lower_in);
}

static PyObject *get_y_upper_in(PyObject *self, PyObject *args) {
    return get_Y(batch.y_upper_in);
}

static PyObject *get_y_lower_out(PyObject *self, PyObject *args) {
    return get_Y(batch.y_lower_out);
}

static PyObject *get_y_upper_out(PyObject *self, PyObject *args) {
    return get_Y(batch.y_upper_out);
}