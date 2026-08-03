#include <cuda.h>
#include <stdio.h>

static int report_cuda_error(const char *operation, CUresult status) {
    const char *name = NULL;
    const char *description = NULL;

    (void)cuGetErrorName(status, &name);
    (void)cuGetErrorString(status, &description);
    (void)fprintf(
        stderr,
        "%s failed: %s (%d): %s\n",
        operation,
        name != NULL ? name : "CUDA_ERROR_UNKNOWN",
        (int)status,
        description != NULL ? description : "no CUDA error description"
    );
    return 1;
}

int main(void) {
    CUresult status = cuInit(0);
    if (status != CUDA_SUCCESS) {
        return report_cuda_error("cuInit", status);
    }

    int device_count = -1;
    status = cuDeviceGetCount(&device_count);
    if (status != CUDA_SUCCESS) {
        return report_cuda_error("cuDeviceGetCount", status);
    }
    if (device_count != 1) {
        (void)fprintf(
            stderr,
            "cuDeviceGetCount returned %d; expected exactly one isolated device\n",
            device_count
        );
        return 1;
    }

    int driver_version = 0;
    status = cuDriverGetVersion(&driver_version);
    if (status != CUDA_SUCCESS) {
        return report_cuda_error("cuDriverGetVersion", status);
    }

    (void)printf(
        "CUDA_DRIVER_API_OK driver_version=%d device_count=%d\n",
        driver_version,
        device_count
    );
    return 0;
}
