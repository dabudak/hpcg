/**
 * @file laik_debug.cpp
 * @brief Implementation of debug functions
 * @version 1.1
 * @date 2023-10-13
 *
 * @copyright Copyright (c) 2023
 *
 */

/*
    Includes
*/

#include <cstring>

#include "laik_debug.hpp"

/*
    Includes
*/

/*
    Implementation of debug functions
*/
/**
 * @brief Compare two double values x and y
 *
 * @param[in] x value
 * @param[in] y value
 * @param[in] doIO print
 * @param[in] curIndex index to the allocation buffer
 */
void compare2(double x, double y, bool doIO, allocation_int_t curIndex)
{
    double delta = std::abs(x - y);

    if (doIO) printf("map_l2a index: %lld\t| %.10f - %.10f | = %.10f\n", curIndex, x, y, delta);

    if (delta != 0.0)
    {
        if (doIO) printf("Difference is not tolerated: %.20f\nindex of map_l2a: %lld\n", delta, curIndex);
        exit_hpcg_run("delta does not equal 0!", false);
    }
}

/**
 * @brief Compare the two vectors x and y.
 *
 * @param[in] x vector
 * @param[in] y vector
 * @param[in] doIO print
 */
void compareResult(Vector &x, Laik_Blob *y, bool doIO)
{
    assert(x.localLength >= y->localLength); // Test vector lengths

    double *xv = x.values;
    double *yv;
    laik_get_map_1d(y->values, 0, (void **)&yv, 0);

    local_int_t length = y->localLength;

    for (local_int_t i = 0; i < length; i++)
    {
        double delta = std::abs(xv[i] - yv[i]);
        if (doIO) printf("xv[%d]=%.10f\tyv_blob[%d]=%.10f\n", i, xv[i], i, yv[i]);
        if (delta != 0)
        {
            if (doIO) printf("Difference is not tolerated: %.20f\n", delta);
            exit_hpcg_run("delta does not equal 0!", false);
        }
    }

    if (doIO) printf("Compare done\n");
}

/**
 * @brief Print the vector
 *
 * @param[in] x vector to be printed
 */
void printResultVector(Vector &x)
{
    if (laik_myid(world) == 0)
    {
        HPCG_fout << "\n\nPrint result of vector\n";

        double *xv = x.values;
        size_t length = x.localLength;

        HPCG_fout << "Length = " << to_string(length) << "\n";

        for (size_t i = 0; i < length; i++)
            HPCG_fout << "xv[" << to_string(i) << "]=" << to_string(xv[i]) << "\n";

        HPCG_fout << "\nEnd of printing result of vector\n\n";
    }
}

/**
 * @brief Print the Laik Vector
 *
 * @param[in] x laik vector to be printed
 */
void printResultLaikVector(Laik_Blob *x)
{
    if (laik_myid(world) == 0)
        printf("\n\nPrint result of Laik-vector\n");

    double *xv;
    laik_get_map_1d(x->values, 0, (void **)&xv, 0);

    size_t localLength = x->localLength;

    if (laik_myid(world) == 0)
    {
        printf("localLength = %ld\n", localLength);
        for (size_t i = 0; i < localLength; i++)
            printf("xv[%ld]=%.10f\n", i, xv[i]);
        printf("\nEnd of printing result of vector\n\n");
    }
}

void printSPM(SparseMatrix *spm, int coarseLevel)
{

    HPCG_fout << "\n##################### Global stats #####################\n\n";

    HPCG_fout << "\nTotal # of rows " << spm->totalNumberOfRows
              << std::endl
              << "\nTotal # of nonzeros " << spm->totalNumberOfNonzeros
              << std::endl;

    HPCG_fout << "\n##################### Local stats #####################\n\n";
    HPCG_fout << "\nLocal # of rows " << spm->localNumberOfRows
              << std::endl
              << "\nLocal # of nonzeros " << spm->localNumberOfNonzeros
              << std::endl;
    HPCG_fout << "\n##################### Mapping of rows #####################\n\n";
    HPCG_fout << "\nLocal-to-global Map\n";
    HPCG_fout << "Local\tGlobal\n";
    for (int c = 0; c < spm->localToGlobalMap.size(); c++)
        HPCG_fout << c << "\t\t" << spm->localToGlobalMap[c] << std::endl;

    if (spm->geom->rank != 0)
    {
        std::cout << "\n##################### My RANK (" << spm->geom->rank << ") #####################\n\n";
        std::cout << "\n##################### Global stats #####################\n\n";

        std::cout << "\nTotal # of rows " << spm->totalNumberOfRows
                  << std::endl
                  << "\nTotal # of nonzeros " << spm->totalNumberOfNonzeros
                  << std::endl;

        std::cout << "\n##################### Local stats #####################\n\n";
        std::cout << "\nLocal # of rows " << spm->localNumberOfRows
                  << std::endl
                  << "\nLocal # of nonzeros " << spm->localNumberOfNonzeros
                  << std::endl;

        std::cout << "\n##################### Mapping of rows #####################\n\n";
        std::cout << "\nLocal-to-global Map\n";
        std::cout << "Local\tGlobal\n";
        for (int c = 0; c < spm->localToGlobalMap.size(); c++)
            std::cout << c << "\t\t" << spm->localToGlobalMap[c] << std::endl;
        std::cout << "\n##################### My RANK (" << spm->geom->rank << ") END #####################\n\n";
    }

    return;
}

void printSPM_val(SparseMatrix &A)
{
#ifdef REPARTITION
    global_int_t nx = A.geom->nx;
    global_int_t ny = A.geom->ny;
    global_int_t nz = A.geom->nz;
    global_int_t gnx = A.geom->gnx;
    global_int_t gny = A.geom->gny;
    global_int_t gnz = A.geom->gnz;
    global_int_t gix0 = A.geom->gix0;
    global_int_t giy0 = A.geom->giy0;
    global_int_t giz0 = A.geom->giz0;

    const char *nonzerosInRow; laik_get_map_1d(A.nonzerosInRow_d, 0, (void **)&nonzerosInRow, 0);
    const double *matrixValues; laik_get_map_1d(A.matrixValues_d, 0, (void **)&matrixValues, 0);
    const double *matrixDiagonal; laik_get_map_1d(A.matrixDiagonal_d, 0, (void **)&matrixDiagonal, 0);
    double entry_val = 0.0; double entry_dia = 0.0;

    std::string debug{""};
    for (local_int_t iz = 0; iz < nz; iz++)
    {
        global_int_t giz = giz0 + iz;
        for (local_int_t iy = 0; iy < ny; iy++)
        {
            global_int_t giy = giy0 + iy;
            for (local_int_t ix = 0; ix < nx; ix++)
            {
                global_int_t gix = gix0 + ix; local_int_t currentLocalRow = iz * nx * ny + iy * nx + ix;
                global_int_t currentGlobalRow = giz * gnx * gny + giy * gnx + gix;

                debug += "Current Local Row (" + std::to_string(currentLocalRow) + ") "
                      + "Current Global Row (" + std::to_string(currentGlobalRow) + ") "
                      + "cur_nnz (" + std::to_string(nonzerosInRow[currentLocalRow]) + ") ";
                debug += "\nUsed Matrix values: ";
                uint64_t currentValuePointer_index = -1;
                global_int_t currentIndexPointerG_index = -1;
                for (int sz = -1; sz <= 1; sz++)
                {
                    if (giz + sz > -1 && giz + sz < gnz)
                    {
                        for (int sy = -1; sy <= 1; sy++)
                        {
                            if (giy + sy > -1 && giy + sy < gny)
                            {
                                for (int sx = -1; sx <= 1; sx++)
                                {
                                    if (gix + sx > -1 && gix + sx < gnx)
                                    {
                                        global_int_t curcol = currentGlobalRow + sz * gnx * gny + sy * gnx + sx;
                                        if (curcol == currentGlobalRow)
                                        {
                                            debug += std::to_string(matrixValues[map_l2a_A(A, currentLocalRow) * numberOfNonzerosPerRow + ++currentValuePointer_index]) + " [dia ";
                                            debug += std::to_string(matrixDiagonal[map_l2a_A(A, currentLocalRow)]) + "], ";
                                        }
                                        else
                                        {
                                            debug += std::to_string(matrixValues[map_l2a_A(A, currentLocalRow) * numberOfNonzerosPerRow + ++currentValuePointer_index]) + ", ";
                                        }
                                        entry_val += matrixValues[map_l2a_A(A, currentLocalRow) * numberOfNonzerosPerRow + currentValuePointer_index];
                                        entry_dia += matrixDiagonal[map_l2a_A(A, currentLocalRow)];
                                        currentIndexPointerG_index++;
                                    }
                                }
                            }
                        }
                    }
                }
                debug += "\n\n";
            }
        }
    }
    HPCG_fout << "Cur entry val sum " << entry_val << " Cur entry dia sum " << entry_dia << "\n";
    HPCG_fout << debug;
#endif
}

void print_HPCG_PARAMS(HPCG_Params params, bool doIO)
{
    if (!doIO)
        return;

    std::string param{""};
    param += "HPCG PARAMETERS\n";
    param += "nx " + std::to_string(params.nx) + "\n";
    param += "ny " + std::to_string(params.ny) + "\n";
    param += "nz " + std::to_string(params.nz) + "\n";
    param += "runningTime " + std::to_string(params.runningTime) + "\n";
    param += "comm_rank " + std::to_string(params.comm_rank) + "\n";
    param += "comm_size " + std::to_string(params.comm_size) + "\n";
    param += "numThreads " + std::to_string(params.numThreads) + "\n";
    param += "npx " + std::to_string(params.npx) + "\n";
    param += "npy " + std::to_string(params.npy) + "\n";
    param += "npz " + std::to_string(params.npz) + "\n";
    param += "pz " + std::to_string(params.pz) + "\n";
    param += "zl " + std::to_string(params.zl) + "\n";
    param += "zu " + std::to_string(params.zu) + "\n";
    printf("%s\n", param.data());
}

void print_GEOMETRY(Geometry *geom, bool doIO)
{
    if (!doIO)
        return;

    std::string param{""};
    param += "GEOMETRY\n";
    param += "rank " + std::to_string(geom->rank) + "\n";
    param += "size " + std::to_string(geom->size) + "\n";
    param += "numThreads " + std::to_string(geom->numThreads) + "\n";
    param += "nx " + std::to_string(geom->nx) + "\n";
    param += "ny " + std::to_string(geom->ny) + "\n";
    param += "nz " + std::to_string(geom->nz) + "\n";
    param += "npx " + std::to_string(geom->npx) + "\n";
    param += "npy " + std::to_string(geom->npy) + "\n";
    param += "npz " + std::to_string(geom->npz) + "\n";
    printf("%s\n", param.data());
}

void print_LaikBlob(Laik_Blob *x)
{
    double *xv;
    laik_get_map_1d(x->values, 0, (void **)&xv, 0);
    std::string n = x->name ? x->name : (char *)"unnamed";
    for (int i = 0; i < x->localLength; i++)
        printf("laik_blob (%s): %f\n", n.data(), xv[i]);
}

void exit_hpcg_run(const char *msg, bool wait)
{
    int rank = 0;
    if (world)
        rank = laik_myid(world);
    if (rank == 0)
    {
        if (strcmp(msg, "") != 0)
            printf("\n\n####### %s\n####### Debug DONE -> Exiting #######\n", msg);
        else
            printf("\n\n####### Debug DONE -> Exiting #######\n");
    }
    if (wait)
        while (1)
            ;
    if (hpcg_instance)
    {
        laik_finalize(hpcg_instance);
        hpcg_instance = nullptr;
        world = nullptr;
    }
    exit(0);
}
