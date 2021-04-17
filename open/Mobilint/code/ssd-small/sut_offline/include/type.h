#ifndef MACCEL_TYPE_H_
#define MACCEL_TYPE_H_

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace maccel_type {
    enum class CollaborationModel {
        Unified, Separated, Undefined
    };
    
    enum class CoreName {
        CentralCore,
        Core1, Core2, Core3, Core4,
        Undefined
    };

    enum class CoreStatus {
        Running, Stopped, Inferencing
    };

    enum class MemoryModel {
        ILDD, BD, 
        Undefined
    };

    enum class PeripheralName {
        STATUS_REG, CONFIG_REG,
        IMEM, LMEM, DMEM, DDR,
        BMEM, CLMEM, DLMEM,
        Undefined
    };

    enum class SchedulePolicy {
        FIFO, LIFO, ByPriority,
        Undefined
    };

    enum class LatencySetPolicy {
        Auto, Manual
    };

    enum class MaintenancePolicy {
        DropExpired,
        Undefined
    };

    /**
     * @brief Mobilint Accelerator 
     * 
     */
    typedef struct Payload {
        uint64_t offset = 0;
        uint8_t* data = nullptr;
        uint64_t size = 0;
    } Payload;

    typedef struct ModelPayload {
        std::string path;
        PeripheralName peripheral;
    } ModelPayload;

    /**
     * @struct QueueBlock
     * Data Structure for waiting Queue
     * 
     * @var QueueBlock::id
     * The id of the request
     * @var QueueBlock::priority
     * The priority of the request (Bigger is more important)
     * @var QueueBlock::core_name
     * The Core Preference of the request
     * @var QueueBlock::rb
     * Information data structure for uploading data
     */
    typedef struct RequestBlock {
        uint64_t id = 0;

        // Information about preferences such as priority, and specific core
        // to process.
        // NOTE: IT DOES NOT GUARANTEE the action as per preferences.
        uint64_t priority = 0;

        std::vector<Payload> payload;
        std::vector<Payload> recv_payload;
    } RequestBlock;

    class BinConfig;
    class V1BinConfig;
    class V2BinConfig;
    class Model;

    class ModelBuilder {
    private:
        std::string mNickname;
        std::vector<BinConfig*> mBinConfig;
        CollaborationModel mCollaborationModel;
        SchedulePolicy mSchedulePolicy;
        LatencySetPolicy mLatencySetPolicy;
        int mCollaborationLeaderSeq = 0;

    public:
        ModelBuilder() = default;
        ~ModelBuilder() = default;

        Model build();
        ModelBuilder& setNickname(std::string nickname);

        ModelBuilder& setBinConfig(V1BinConfig* bin_config);
        ModelBuilder& setBinConfig(V2BinConfig* bin_config);
        ModelBuilder& setBinConfig(std::vector<V1BinConfig *> bin_configs);
        ModelBuilder& setBinConfig(std::vector<V2BinConfig *> bin_configs);

        ModelBuilder& setCollaborationModel(
            CollaborationModel collaboration_model);
        ModelBuilder& setCollaborationLeader(int seq);
        ModelBuilder& setSchedulePolicy(SchedulePolicy policy);
        ModelBuilder& setLatencySetPolicy(LatencySetPolicy policy);
    };

    class Model {
    private:
        class Statistics;

        std::string mNickname;
        std::vector<BinConfig*> mBinConfig;
        CollaborationModel mCollaborationModel = CollaborationModel::Undefined;
        SchedulePolicy mSchedulePolicy;
        LatencySetPolicy mLatencySetPolicy;
        MaintenancePolicy mMaintenancePolicy;
        uint64_t mLatencyConsumed;
        uint64_t mLatencyFinished;

        std::shared_ptr<Statistics> mStatistics;
    public:
        Model();
        ~Model();

        bool setNickname(std::string nickname);
        bool setBinConfig(std::vector<BinConfig*> bin_config);
        bool setCollaborationModel(CollaborationModel collaboration_model); 
        bool setSchedulePolicy(SchedulePolicy policy);
        bool setLatencySetPolicy(LatencySetPolicy policy);
        bool setMaintenancePolicy(MaintenancePolicy policy);
        bool setLatency(uint64_t latency_consumed, uint64_t latency_finish);

        std::string getNickname() const;
        std::vector<BinConfig*> getBinConfig() const;
        CollaborationModel getCollaborationModel() const; 
        SchedulePolicy getSchedulePolicy() const;
        LatencySetPolicy getLatencySetPolicy() const;
        MaintenancePolicy getMaintenancePolicy() const;
        uint64_t getLatencyConsumed() const;
        uint64_t getLatencyFinished() const;
        std::shared_ptr<Statistics> getStatistics() const;

        bool operator==(const Model &other) const;
    };

    class BinConfig {
    protected:
        CoreName mCoreName = CoreName::Undefined;
        uint64_t mLatencyConsumed = 0;
        uint64_t mLatencyFinished = 0;
    public:
        virtual ~BinConfig() = 0;
        virtual CoreName getCoreName() = 0;
        virtual uint64_t getLatencyConsumed();
        virtual uint64_t getLatencyFinished();
        virtual std::vector<ModelPayload> getRequestBlocks() = 0;
    };

    class V1BinConfig : public BinConfig {
    private:
        std::string mPathImem = "";
        std::string mPathLmem = "";
        std::string mPathDmem = "";
        std::string mPathDDR = "";
    public:
        V1BinConfig() = delete;
        V1BinConfig(
            CoreName core_name, 
            std::string imem, std::string lmem,
            std::string dmem, std::string ddr, 
            uint64_t latency_consumed, uint64_t latency_finish);
        ~V1BinConfig() override = default;

        CoreName getCoreName();
        std::vector<ModelPayload> getRequestBlocks();
    };

    class V2BinConfig : public BinConfig {
    private:
        std::string mPathBram = "";
        std::string mPathDmem = "";
    public:
        V2BinConfig() = delete;
        V2BinConfig(
            CoreName core_name,
            std::string bram, std::string dmem,
            uint64_t latency_consumed, uint64_t latency_finish);
        ~V2BinConfig() override = default;

        CoreName getCoreName();
        std::vector<ModelPayload> getRequestBlocks();
    };
};

#endif