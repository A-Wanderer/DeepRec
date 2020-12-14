#include "gtest/gtest.h"
#include "odl_processor/serving/model_config.h"

namespace tensorflow {
namespace processor {

class ModelConfigTest : public ::testing::Test {
 protected:
  ModelConfigTest() {}
};

TEST_F(ModelConfigTest, ShouldReturnInvalidWhenEmptyInput) {
  ModelConfig* model_config = nullptr;
  EXPECT_FALSE(ModelConfigFactory::Create("", &model_config).ok());
  EXPECT_EQ(nullptr, model_config);
}

TEST_F(ModelConfigTest, ShouldSuccessWhenOssAndRedis) {
const std::string oss_and_redis_config = " \
  { \
    \"inter_op_parallelism_threads\" : 4, \
    \"intra_op_parallelism_threads\" : 2, \
    \"init_timeout_minutes\" : 1, \
    \"signature_name\": \"tensorflow_serving\", \
    \"checkpoint_dir\" : \"oss://test_ckpt/1\", \
    \"savedmodel_dir\" : \"oss://test_savedmodel/1\", \
    \"feature_store_type\" : \"cluster_redis\", \
    \"redis_url\" :\"test_url\",  \
    \"redis_password\" :\"test_password\", \
    \"read_thread_num\" : 2, \
    \"update_thread_num\":1, \
    \"model_store_type\": \"oss\", \
    \"oss_endpoint\": \"test.endpoint\", \
    \"oss_access_id\" : \"test_id\", \
    \"oss_access_key\" : \"test_key\" \
  }";

  ModelConfig* config = nullptr;
  //EXPECT_TRUE(ModelConfigFactory::Create(oss_and_redis_config.c_str(), &config).ok());
  auto s = ModelConfigFactory::Create(oss_and_redis_config.c_str(), &config);
  EXPECT_EQ(s.error_message(), "");
  
  EXPECT_EQ(4, config->inter_threads);
  EXPECT_EQ(2, config->intra_threads);
  EXPECT_EQ("tensorflow_serving", config->signature_name);
  EXPECT_EQ(
      "oss://test_ckpt\x1id=test_id\x2key=test_key\x2host=test.endpoint/1",
      config->checkpoint_dir);
  EXPECT_EQ(
      "oss://test_savedmodel\x1id=test_id\x2key=test_key\x2host=test.endpoint/1",
      config->savedmodel_dir);
  EXPECT_EQ("cluster_redis", config->feature_store_type);
  EXPECT_EQ("test_url", config->redis_url);
  EXPECT_EQ("test_password", config->redis_password);
  EXPECT_EQ(2, config->read_thread_num);
  EXPECT_EQ(1, config->update_thread_num);
  EXPECT_EQ("oss", config->model_store_type);
  EXPECT_EQ("test.endpoint", config->oss_endpoint);
  EXPECT_EQ("test_id", config->oss_access_id);
  EXPECT_EQ("test_key", config->oss_access_key);
}

TEST_F(ModelConfigTest, ShouldFailureWhenSignatureNameEmpty) {
const std::string oss_and_redis_config = " \
  { \
    \"inter_op_parallelism_threads\" : 4, \
    \"intra_op_parallelism_threads\" : 2, \
    \"init_timeout_minutes\" : 1, \
    \"signature_name\": \"\", \
    \"checkpoint_dir\" : \"oss://test_ckpt/1\", \
    \"savedmodel_dir\" : \"oss://test_savedmodel/1\", \
    \"feature_store_type\" : \"cluster_redis\", \
    \"redis_url\" :\"test_url\",  \
    \"redis_password\" :\"test_password\", \
    \"read_thread_num\" : 2, \
    \"update_thread_num\":1, \
    \"model_store_type\": \"oss\", \
    \"oss_endpoint\": \"test.endpoint\", \
    \"oss_access_id\" : \"test_id\", \
    \"oss_access_key\" : \"test_key\" \
  }";

  ModelConfig* config = nullptr;
  EXPECT_EQ(error::Code::INVALID_ARGUMENT,
      ModelConfigFactory::Create(oss_and_redis_config.c_str(), &config).code());
  EXPECT_EQ(4, config->inter_threads);
  EXPECT_EQ(2, config->intra_threads);
  EXPECT_EQ("", config->signature_name);
}

TEST_F(ModelConfigTest, ShouldFailedWhenOssDirAndModelStroreTypeMismatch) {
const std::string oss_and_redis_config = " \
  { \
    \"inter_op_parallelism_threads\" : 4, \
    \"intra_op_parallelism_threads\" : 2, \
    \"init_timeout_minutes\" : 1, \
    \"signature_name\": \"tensorflow_serving\", \
    \"checkpoint_dir\" : \"hdfs://test_ckpt/1\", \
    \"savedmodel_dir\" : \"hdfs://test_savedmodel/1\", \
    \"feature_store_type\" : \"cluster_redis\", \
    \"redis_url\" :\"test_url\",  \
    \"redis_password\" :\"test_password\", \
    \"read_thread_num\" : 2, \
    \"update_thread_num\":1, \
    \"model_store_type\": \"oss\", \
    \"oss_endpoint\": \"test.endpoint\", \
    \"oss_access_id\" : \"test_id\", \
    \"oss_access_key\" : \"test_key\" \
  }";

  ModelConfig* config = nullptr;
  EXPECT_EQ(error::Code::INVALID_ARGUMENT,
      ModelConfigFactory::Create(oss_and_redis_config.c_str(), &config).code());
  EXPECT_EQ(4, config->inter_threads);
  EXPECT_EQ(2, config->intra_threads);
  EXPECT_EQ("tensorflow_serving", config->signature_name);
  EXPECT_EQ("hdfs://test_ckpt/1", config->checkpoint_dir);
  EXPECT_EQ("hdfs://test_savedmodel/1", config->savedmodel_dir);
  EXPECT_EQ("cluster_redis", config->feature_store_type);
  EXPECT_EQ("test_url", config->redis_url);
  EXPECT_EQ("test_password", config->redis_password);
  EXPECT_EQ(2, config->read_thread_num);
  EXPECT_EQ(1, config->update_thread_num);
  EXPECT_EQ("oss", config->model_store_type);
  EXPECT_EQ("", config->oss_endpoint);
  EXPECT_EQ("", config->oss_access_id);
  EXPECT_EQ("", config->oss_access_key);
}

TEST_F(ModelConfigTest, ShouldSuccessWhenOssDirAndNativeTFFeatureStoreType) {
const std::string oss_and_redis_config = " \
  { \
    \"inter_op_parallelism_threads\" : 4, \
    \"intra_op_parallelism_threads\" : 2, \
    \"init_timeout_minutes\" : 1, \
    \"signature_name\": \"tensorflow_serving\", \
    \"checkpoint_dir\" : \"oss://test_ckpt/1\", \
    \"savedmodel_dir\" : \"oss://test_savedmodel/1\", \
    \"feature_store_type\" : \"tensorflow\", \
    \"redis_url\" :\"test_url\",  \
    \"redis_password\" :\"test_password\", \
    \"read_thread_num\" : 2, \
    \"update_thread_num\":1, \
    \"model_store_type\": \"oss\", \
    \"oss_endpoint\": \"test.endpoint\", \
    \"oss_access_id\" : \"test_id\", \
    \"oss_access_key\" : \"test_key\" \
  }";

  ModelConfig* config = nullptr;
  EXPECT_TRUE(
      ModelConfigFactory::Create(oss_and_redis_config.c_str(), &config).ok());
  EXPECT_EQ(4, config->inter_threads);
  EXPECT_EQ(2, config->intra_threads);
  EXPECT_EQ("tensorflow_serving", config->signature_name);
  EXPECT_EQ(
      "oss://test_ckpt\x1id=test_id\x2key=test_key\x2host=test.endpoint/1",
      config->checkpoint_dir);
  EXPECT_EQ(
      "oss://test_savedmodel\x1id=test_id\x2key=test_key\x2host=test.endpoint/1",
      config->savedmodel_dir);
  EXPECT_EQ("tensorflow", config->feature_store_type);
  EXPECT_EQ("", config->redis_url);
  EXPECT_EQ("", config->redis_password);
  EXPECT_EQ(1, config->read_thread_num);
  EXPECT_EQ(1, config->update_thread_num);
  EXPECT_EQ("oss", config->model_store_type);
  EXPECT_EQ("test.endpoint", config->oss_endpoint);
  EXPECT_EQ("test_id", config->oss_access_id);
  EXPECT_EQ("test_key", config->oss_access_key);
}

} // processor
} // tensorflow

