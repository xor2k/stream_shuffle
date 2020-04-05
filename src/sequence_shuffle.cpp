
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <algorithm>

// Forward function declaration.
static PyObject *create_shuffle(PyObject *self, PyObject *args);
static PyObject *get_batch_count(PyObject *self, PyObject *args);

static PyObject *create_batch(PyObject *self, PyObject *args);
static PyObject *get_X(PyObject *self, PyObject *args);
static PyObject *get_y_lower_in(PyObject *self, PyObject *args);
static PyObject *get_y_upper_in(PyObject *self, PyObject *args);
static PyObject *get_y_lower_out(PyObject *self, PyObject *args);
static PyObject *get_y_upper_out(PyObject *self, PyObject *args);

// Boilerplate: method list.
static PyMethodDef methods[] = {
    { "create_shuffle", create_shuffle, METH_VARARGS, "Doc string."},
    { "get_batch_count", get_batch_count, METH_NOARGS, "Doc string."},

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
#define MASK_TOKEN -2

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
    // srand (time(NULL));
    srand(0);

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
    pixel_type* data;
    size_t video_count;

    size_t* video_pixel_counts;
    size_t* video_pixel_offsets;

    label_type* labels_in;
    label_type* labels_out;
} data;

static struct shuffle_struct {
    size_t* shuffled_ids;
} shuffle;

static struct options_struct {
    size_t batch_size;
    size_t aggregation_length;
    size_t batch_count;
} options;

static struct batch_struct {
    size_t batch_id;
    size_t frame_count;
    size_t* aggregation_pixel_counts;

    pixel_type* X;
    label_type* y_lower_in;
    label_type* y_upper_in;
    label_type* y_lower_out;
    label_type* y_upper_out;
} batch;

static struct epoch_state_struct {
    size_t current_batch_id;
} epoch_state;

/*****************************************************************************
 * shuffle                                                                   *
 *****************************************************************************/
static PyObject *create_shuffle(PyObject *self, PyObject *args) {
    memset(&data, 0, sizeof(struct data_struct));
    memset(&shuffle, 0, sizeof(struct shuffle_struct));
    memset(&batch, 0, sizeof(struct batch_struct));
    memset(&options, 0, sizeof(struct options_struct));
    memset(&epoch_state, 0, sizeof(struct epoch_state_struct));

    PyArrayObject *py_data;
    PyArrayObject *py_offsets;
    PyArrayObject *py_labels_in;
    PyArrayObject *py_labels_out;

    if (!PyArg_ParseTuple(
        args, "kkO!O!O!O!",
        &options.batch_size,
        &options.aggregation_length,
        &PyArray_Type, &py_data,
        &PyArray_Type, &py_offsets,
        &PyArray_Type, &py_labels_in,
        &PyArray_Type, &py_labels_out
    )) {
        return NULL;
    }

    // initialize data
    data.video_count = PyArray_SIZE(py_offsets);
    size_t data_length = PyArray_SIZE(py_data);
    data.data = (pixel_type*)PyArray_DATA(py_data);

    data.video_pixel_counts = (size_t*)realloc(
        data.video_pixel_counts, sizeof(size_t)*data.video_count
    );

    data.video_pixel_offsets = (size_t*)realloc(
        data.video_pixel_offsets, sizeof(size_t)*data.video_count
    );

    data.labels_in = (npy_float32*)realloc(
        data.labels_in, sizeof(npy_float32)*data.video_count
    );

    data.labels_out = (npy_float32*)realloc(
        data.labels_out, sizeof(npy_float32)*data.video_count
    );

    for(size_t i=0; i<data.video_count; i++){
        data.video_pixel_counts[i] = i == data.video_count - 1 ?
            data_length - offsets(i) : offsets(i+1)-offsets(i);
        data.video_pixel_offsets[i] = offsets(i);
        data.labels_in[i] = labels_in(i);
        data.labels_out[i] = labels_out(i);
    }

    // initialize shuffle
    free(shuffle.shuffled_ids);
    shuffle.shuffled_ids = make_shuffled_ids(data.video_count);
    options.batch_count = (size_t)ceil((double) data.video_count / (
        (double) options.aggregation_length *
        (double) options.batch_size
    ));

    // initialize batch
    batch.batch_id = 0;
    epoch_state.current_batch_id = 0;

    batch.aggregation_pixel_counts = (size_t*) realloc(
        batch.aggregation_pixel_counts,
        sizeof(size_t) * options.batch_size
    );

    Py_RETURN_NONE;

}

static PyObject *create_batch(PyObject *self, PyObject *args) {
    if(batch.batch_id == options.batch_count){
        Py_RETURN_NONE;
    }

    memset(
        batch.aggregation_pixel_counts, 0, sizeof(size_t)*options.batch_size
    );

    batch.batch_id = epoch_state.current_batch_id++;

    size_t batch_first_video_index =
        data.video_count / options.batch_count * batch.batch_id;

    size_t batch_last_video_index = std::min(
        batch_first_video_index +
        options.batch_size * options.aggregation_length,
        data.video_count
    );

    size_t longest_aggregation_pixel_count = 0;
    size_t current_aggregation_pixel_count = 0;
    for(size_t i = batch_first_video_index; i < batch_last_video_index; i++){
        if((i-batch_first_video_index)%options.aggregation_length == 0){
            current_aggregation_pixel_count = 0;
        }
        size_t shuffled_id = shuffle.shuffled_ids[i];
        current_aggregation_pixel_count+=data.video_pixel_counts[shuffled_id];

        longest_aggregation_pixel_count = std::max(
            longest_aggregation_pixel_count,
            current_aggregation_pixel_count
        );
    }

    batch.X = (pixel_type*)realloc(
        batch.X, sizeof(pixel_type)*
        options.batch_size*longest_aggregation_pixel_count
    );

    auto y_byte_count = sizeof(pixel_type)*options.batch_size*
        longest_aggregation_pixel_count/FRAME_PIXEL_COUNT;

    auto y_ptrs = {
        &batch.y_lower_in, &batch.y_lower_out,
        &batch.y_upper_in, &batch.y_upper_out
    };

    for(auto ptr : y_ptrs){
        *ptr = (label_type*)realloc(*ptr, y_byte_count);
    }

    size_t aggregation_lower_bound_in = 0;
    size_t aggregation_lower_bound_out = 0;
    size_t aggregation_upper_bound_in = 0;
    size_t aggregation_upper_bound_out = 0;
    size_t aggregation_pixel_count = 0;
    size_t aggregation_current_index = 0;
    for(size_t i = batch_first_video_index; i < batch_last_video_index; i++){
        size_t i_rel = (i-batch_first_video_index)%options.aggregation_length;
        if(i_rel == 0){
            aggregation_current_index =
                i == batch_first_video_index ? 0 : aggregation_current_index+1;

            aggregation_lower_bound_in = 0;
            aggregation_lower_bound_out = 0;
            aggregation_upper_bound_in = 0;
            aggregation_upper_bound_out = 0;

            aggregation_pixel_count = 0;
        }
        size_t shuffled_id = shuffle.shuffled_ids[i];

        auto label_in = data.labels_in[shuffled_id];
        auto label_out = data.labels_out[shuffled_id];

        auto video_pixel_count = data.video_pixel_counts[shuffled_id];
        auto video_frame_count = video_pixel_count / FRAME_PIXEL_COUNT;
        auto data_pixel_offset = data.video_pixel_offsets[shuffled_id];

        int mode = rand() % 4;
        size_t do_reverse = mode == 2 || mode == 3;

        aggregation_upper_bound_in += do_reverse ? label_out : label_in;
        aggregation_upper_bound_out += do_reverse ? label_in : label_out;

        size_t batch_pixel_offset =
            aggregation_pixel_count +
            aggregation_current_index * longest_aggregation_pixel_count;

        for(auto tmp : y_ptrs){
            auto ptr = *tmp + batch_pixel_offset/FRAME_PIXEL_COUNT;
            std::fill(ptr, ptr + video_frame_count,
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
                    &batch.X[batch_pixel_offset],
                    &data.data[data_pixel_offset],

                    video_pixel_count*sizeof(pixel_type)
                );
                break;
            // mirror
            case 1:
                for(size_t j = 0; j < video_frame_count; j++){
                    auto frame_offset = j * FRAME_PIXEL_COUNT;
                    for(size_t k = 0; k < FRAME_WIDTH; k++){
                        for(size_t l = 0; l < FRAME_HEIGHT; l++){
                            batch.X[
                                batch_pixel_offset + frame_offset +
                                l * FRAME_WIDTH + k
                            ] = data.data[
                                data_pixel_offset + frame_offset +
                                l * FRAME_WIDTH + (FRAME_WIDTH - 1 - k)
                            ];
                        }
                    }
                }
                break;
            // reverse
            case 2:
                for(size_t j = 0; j < video_frame_count; j++){
                    memcpy(
                        &batch.X[
                            batch_pixel_offset + j * FRAME_PIXEL_COUNT
                        ],
                        &data.data[
                            data_pixel_offset
                            + (video_frame_count - 1 - j) * FRAME_PIXEL_COUNT
                        ],

                        FRAME_PIXEL_COUNT*sizeof(pixel_type)
                    );
                }
                break;
            // reverse and mirror
            case 3:
                for(size_t j = 0; j < video_frame_count; j++){
                    for(size_t k = 0; k < FRAME_WIDTH; k++){
                        for(size_t l = 0; l < FRAME_HEIGHT; l++){
                            batch.X[
                                batch_pixel_offset + (j * FRAME_PIXEL_COUNT) +
                                l * FRAME_WIDTH + k
                            ] = data.data[
                                data_pixel_offset + (video_frame_count-1-j)
                                * FRAME_PIXEL_COUNT +
                                l * FRAME_WIDTH + (FRAME_WIDTH - 1 - k)
                            ];
                        }
                    }
                }
                break;
        }
        aggregation_lower_bound_in += label_in;
        aggregation_lower_bound_out += label_out;

        aggregation_pixel_count += video_pixel_count;

        batch.aggregation_pixel_counts[aggregation_current_index] =
            aggregation_pixel_count;
    }

    for(size_t i=0; i<options.batch_size; i++){
        size_t aggregation_pixel_count = batch.aggregation_pixel_counts[i];
        size_t aggregation_batch_pixel_offset = i*longest_aggregation_pixel_count;
        size_t aggregation_batch_pixel_offset_next =
            (i+1)*longest_aggregation_pixel_count;

        std::fill(
            batch.X + aggregation_batch_pixel_offset + aggregation_pixel_count,
            batch.X + aggregation_batch_pixel_offset_next,
            MASK_TOKEN
        );

        size_t aggregation_frame_count =
            aggregation_pixel_count / FRAME_PIXEL_COUNT;

        size_t aggregation_batch_frame_offset =
            aggregation_batch_pixel_offset / FRAME_PIXEL_COUNT;

        size_t aggregation_batch_frame_offset_next =
            aggregation_batch_pixel_offset_next / FRAME_PIXEL_COUNT;

        for(auto ptr : y_ptrs){
            std::fill(
                *ptr + aggregation_batch_frame_offset + aggregation_frame_count,
                *ptr + aggregation_batch_frame_offset_next,
                MASK_TOKEN
            );
        }
    }

    batch.frame_count = options.batch_size*longest_aggregation_pixel_count/FRAME_PIXEL_COUNT;

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

static PyObject *get_batch_count(PyObject *self, PyObject *args) {
    return PyLong_FromLongLong(options.batch_count);
}