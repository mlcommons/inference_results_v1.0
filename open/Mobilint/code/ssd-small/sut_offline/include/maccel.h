#ifndef MACCEL_MACCEL_H_
#define MACCEL_MACCEL_H_

#include <memory>
#include <vector>
#include "type.h"

namespace mobilint {
    class Accelerator {
    private:
        class Impl;
        std::unique_ptr<Impl> mImpl;
    
    public:
        Accelerator(uint64_t arch, int dev_no, bool verbose);
        Accelerator() = delete;

        /**
         * Destruct the class.
         */
        ~Accelerator();

        /**
         * @brief Allocate the appropriate memory
         * 
         * @param allocated Vector of allocated memory.
         * @return true Successfully allocated.
         * @return false Unsuccessful allocation.
         */
        bool allocateMemory(
            std::vector<std::pair<void **, uint64_t>> allocated);
        bool allocateMemory(
            std::vector<std::pair<uint8_t **, uint64_t>> allocated);
        bool allocateMemory(
            std::vector<std::pair<int8_t **, uint64_t>> allocated);

        /**
         * @brief Free the memory
         * 
         * @param victim Victim of deallocation.
         * @return true Successfully deallocated.
         * @return false Unsuccessful deallocation.
         */
        bool freeMemory(std::vector<void *> victim);
        bool freeMemory(std::vector<uint8_t *> victim);
        bool freeMemory(std::vector<int8_t *> victim);

        /**
         * Return the initialization status of the device.
         * 
         * @return true if it is initialized, false if not
         */
        bool isInitialized();
        std::vector<maccel_type::CoreName> getSupportedCores();

        bool run(const maccel_type::Model core); 
        bool run(const std::vector<maccel_type::Model>& models);
        bool stop(const maccel_type::Model core, bool immediate);
        bool stop(const std::vector<maccel_type::Model>& models, bool immediate);
        
        maccel_type::CoreStatus getCoreStatus(maccel_type::CoreName core);

        /**
         * Post the inference request and receive request atomically.
         * Both adding Request and Receive Request should be atomic.
         * 
         * @param request A vector array holding a request info.
         * @param receive A vector array holding a receive buffer info.
         */
        uint64_t request(const maccel_type::Model& model, 
            std::vector<maccel_type::Payload>& payload_request, 
            std::vector<maccel_type::Payload>& payload_receive);

        bool receive(uint64_t ticket);

        /**
         * Setup the model into the Accelerator.
         * 
         * @param imem_path Path to imem.bin
         * @param lmem_path Path to lmem.bin
         * @param dmem_path Path to dmem.bin
         * @param weight_path Path to ddr.bin
         * @return 0 if succesful, 1 otherwise.
         */
        bool setModel(std::vector<maccel_type::Model>& models);
        bool setModel(maccel_type::Model& model);

        /*
        TODO:
        getRequests(model)
        setOnCompleteListener(listener)
        rescue()
        */
    };
}

#endif