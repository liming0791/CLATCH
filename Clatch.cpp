#include "Clatch.h"

Clatch::Clatch()
{
    // setting cache and shared modes
	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

    // allocating and transferring triplets and binding to texture object
	uint32_t* d_triplets;
	cudaMalloc(&d_triplets, 2048 * sizeof(uint16_t));
	cudaMemcpy(d_triplets, triplets, 2048 * sizeof(uint16_t), cudaMemcpyHostToDevice);

	cudaChannelFormatDesc chandesc_trip = cudaCreateChannelDesc(
            16, 16, 16, 16, cudaChannelFormatKindUnsigned);
	cudaArray* d_trip_arr;
	cudaMallocArray(&d_trip_arr, &chandesc_trip, 512);
	cudaMemcpyToArray(d_trip_arr, 0, 0, d_triplets, 2048 * sizeof(uint16_t), cudaMemcpyHostToDevice);

	struct cudaResourceDesc resdesc_trip;
	memset(&resdesc_trip, 0, sizeof(resdesc_trip));
	resdesc_trip.resType = cudaResourceTypeArray;
	resdesc_trip.res.array.array = d_trip_arr;

	struct cudaTextureDesc texdesc_trip;
	memset(&texdesc_trip, 0, sizeof(texdesc_trip));
	texdesc_trip.addressMode[0] = cudaAddressModeClamp;
	texdesc_trip.filterMode = cudaFilterModePoint;
	texdesc_trip.readMode = cudaReadModeElementType;
	texdesc_trip.normalizedCoords = 0;

	d_trip_tex = 0;
	cudaCreateTextureObject(&d_trip_tex, &resdesc_trip, &texdesc_trip, nullptr);

}

Clatch::~Clatch()
{
}

void Clatch::compute(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &desc)
{
    // arranging keypoints for PCI transfer
	std::vector<CLATCHKeyPoint> kps;
	for (const auto& kp : keypoints) 
        kps.emplace_back(kp.pt.x, kp.pt.y, kp.size, kp.angle * 3.14159265f / 180.0f);

    // time begin
	//high_resolution_clock::time_point start = high_resolution_clock::now();

    // allocating space for descriptors
	uint64_t* d_desc;
	cudaMalloc(&d_desc, 64 * (int)kps.size());     //cuda malloc

	// allocating and transferring keypoints and binding to texture object
    CLATCHKeyPoint* d_kps;
	cudaMalloc(&d_kps, kps.size() * sizeof(CLATCHKeyPoint));    //cuda malloc
	cudaMemcpy(d_kps, &kps[0], kps.size() * sizeof(CLATCHKeyPoint), cudaMemcpyHostToDevice);

    // allocating and transferring image and binding to texture object
	cudaChannelFormatDesc chandesc_img = 
        cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray* d_img_arr;
	cudaMallocArray(&d_img_arr, &chandesc_img, image.cols, image.rows);     //cuda malloc
	cudaMemcpyToArray(d_img_arr, 0, 0, image.data, image.rows * image.cols, cudaMemcpyHostToDevice);

	struct cudaResourceDesc resdesc_img;
	memset(&resdesc_img, 0, sizeof(resdesc_img));
	resdesc_img.resType = cudaResourceTypeArray;

	resdesc_img.res.array.array = d_img_arr;

	struct cudaTextureDesc texdesc_img;
	memset(&texdesc_img, 0, sizeof(texdesc_img));
	texdesc_img.addressMode[0] = cudaAddressModeClamp;
	texdesc_img.addressMode[1] = cudaAddressModeClamp;
	texdesc_img.filterMode = cudaFilterModePoint;
	texdesc_img.readMode = cudaReadModeElementType;
	texdesc_img.normalizedCoords = 0;

    cudaTextureObject_t d_img_tex=0;
	cudaCreateTextureObject(&d_img_tex, &resdesc_img, &texdesc_img, nullptr);
    // do clatch
	CLATCH(d_img_tex, d_trip_tex, d_kps, static_cast<int>(kps.size()), d_desc);

    // time end
	//high_resolution_clock::time_point end = high_resolution_clock::now();
	// --------------------------------

	//std::cout << std::endl << "CLATCH took " 
    //    << static_cast<double>((end - start).count()) * 1e-3 / static_cast<double>(kps.size()) 
    //    << " us per desc over " << kps.size() << " desc" 
    //    << (kps.size() == 1 ? "." : "s.") << std::endl << std::endl;
	
    // set result
	cudaMemcpy(desc.data, d_desc, 64 * kps.size(), cudaMemcpyDeviceToHost);

    // cuda release
    cudaFree(d_desc);
    cudaFree(d_kps);
    cudaFreeArray(d_img_arr);

	//std::cout << "CUDA reports " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	//long long total = 0;
	//for (size_t i = 0; i < 8 * kps.size(); ++i) total += h_GPUdesc[i];
	//std::cout << "Checksum: " << std::hex << total << std::endl << std::endl;

}
