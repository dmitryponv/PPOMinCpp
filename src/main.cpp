#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>
#include <string>
#include <tuple>


struct ArgList {
    int seed = 0;
    std::string domain = "";
    std::string task = "";
    std::string mode = "train";
    bool resume = false;
    double lr = 1e-3;                 // Increased learning rate
    int episodes = 100;
    int update_every = 20;           // More frequent updates
    double gamma = 0.99;
    double gae_lambda = 0.95;
    int epochs = 5;                  // Fewer epochs per update
    int batch_size = 128;            // Larger batches
    double target_kl = 0.01;
    double ppo_clip_term = 0.2;
    double entropy_weightage = 0.01; // Promote exploration
    double gradient_clip_term = 0.5;
    int eval_every = 250;            // Evaluate more often
    int eval_over = 3;
};

class FFmpegWriter {
    FILE* pipe;
    std::string command;
    int w, h;

public:
    FFmpegWriter(const std::string& filename, int width, int height, int fps)
        : w(width), h(height)
    {
        command = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 "
            "-s " + std::to_string(w) + "x" + std::to_string(h) +
            " -r " + std::to_string(fps) +
            " -i - -an -vcodec libx264 -pix_fmt yuv420p \"" + filename + "\"";

        pipe = _popen(command.c_str(), "wb");
        if (!pipe) throw std::runtime_error("Failed to open FFmpeg pipe.");
    }

    void write_frame(const void* data) {
        fwrite(data, 1, 3 * w * h, pipe);
    }

    int width() const { return w; }
    int height() const { return h; }

    ~FFmpegWriter() {
        if (pipe) _pclose(pipe);
    }
};

void write_rgb_pattern_mp4() {
    const int width = 640;
    const int height = 480;
    const int frames = 30;
    std::vector<uint8_t> frame(width * height * 3);

    // Create a horizontal RGB gradient pattern
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            frame[idx + 0] = static_cast<uint8_t>((x * 255) / width);  // R
            frame[idx + 1] = static_cast<uint8_t>((y * 255) / height); // G
            frame[idx + 2] = static_cast<uint8_t>(255 - (x * 255) / width); // B
        }
    }

    std::string command = "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 640x480 -r 30 "
        "-i - -an -vcodec libx264 -pix_fmt yuv420p test.mp4";

    FILE* pipe = _popen(command.c_str(), "wb");
    if (!pipe) {
        std::cerr << "Failed to open ffmpeg pipe." << std::endl;
        return;
    }

    for (int i = 0; i < frames; ++i) {
        fwrite(frame.data(), 1, frame.size(), pipe);
    }

    fflush(pipe);
    _pclose(pipe);
}


struct TimeStep {
    std::map<std::string, torch::Tensor> observation;
    double reward;
    bool last_flag;
    bool done;
    bool last() const { return last_flag; }
};

//The environment
class GridWorldEnv {
public:
    static constexpr int kGridSize_h = 80;
    static constexpr int kGridSize_v = 25;
    static constexpr int kObsSize = 4;       // agent_x, agent_y, target_x, target_y
    static constexpr int kNumActions = 2;    // up, down, left, right
    int min_dist = 60;

    TimeStep& reset() {

        do {
            agent_x = dist_x(gen);
            agent_y = dist_y(gen);
            target_x = dist_x(gen);
            target_y = dist_y(gen);
        } while ((agent_x == target_x && agent_y == target_y) ||
            ((agent_x - target_x) * (agent_x - target_x) +
                (agent_y - target_y) * (agent_y - target_y)) < min_dist);

        time_step.observation["agent_x"] = torch::tensor(agent_x, torch::kInt32);
        time_step.observation["agent_y"] = torch::tensor(agent_y, torch::kInt32);
        time_step.observation["target_x"] = torch::tensor(target_x, torch::kInt32);
        time_step.observation["target_y"] = torch::tensor(target_y, torch::kInt32);

        time_step.reward = 0.0;
        time_step.done = false;
        time_step.last_flag = false;

        return time_step;
    }

    TimeStep& step(torch::Tensor actions) {
        // Convert actions to CPU and float
        auto acts = actions.to(torch::kCPU).to(torch::kFloat32).squeeze();

        float agent_dx = acts[0].item<float>();
        float agent_dy = acts[1].item<float>();

        int prev_agent_x = agent_x;
        int prev_agent_y = agent_y;

        float prev_dist = std::hypot(agent_x - target_x, agent_y - target_y);

        int new_agent_x = agent_x + static_cast<int>(std::round(agent_dx));
        int new_agent_y = agent_y + static_cast<int>(std::round(agent_dy));

        bool out_of_bounds =
            new_agent_x < 0 || new_agent_x >= kGridSize_h ||
            new_agent_y < 0 || new_agent_y >= kGridSize_v;

        agent_x = std::clamp(new_agent_x, 0, kGridSize_h - 1);
        agent_y = std::clamp(new_agent_y, 0, kGridSize_v - 1);

        float new_dist = std::hypot(agent_x - target_x, agent_y - target_y);

        time_step.observation["agent_x"] = torch::tensor(agent_x, torch::kInt32);
        time_step.observation["agent_y"] = torch::tensor(agent_y, torch::kInt32);
        time_step.observation["target_x"] = torch::tensor(target_x, torch::kInt32);
        time_step.observation["target_y"] = torch::tensor(target_y, torch::kInt32);

        float reward = 0.0f;

        if (out_of_bounds) {
            reward = -1.0f;
            time_step.done = true;
        }
        else if (new_dist <= 1.0f) {
            reward = 1.0f;
            time_step.done = true;
        }
        else {
            reward = (new_dist < prev_dist) ? 0.01f : -0.01f;
            reward += (1.0f / new_dist); // Add inverse distance reward
            time_step.done = false;
        }

        time_step.reward = reward;

        return time_step;
    }

    GridWorldEnv() : gen(rd()), dist_x(0, kGridSize_h - 1), dist_y(0, kGridSize_v - 1) {
        reset();
    }

    void render_text() const {
        system("cls");

        for (int y = 0; y < kGridSize_v; ++y) {
            for (int x = 0; x < kGridSize_h; ++x) {
                if (x == agent_x && y == agent_y) {
                    std::cout << "A";  // Agent
                }
                else if (x == target_x && y == target_y) {
                    std::cout << "T";  // Target
                }
                else {
                    std::cout << ".";
                }
            }
            std::cout << '\n';
        }
    }

    std::vector<uint8_t> render(int& width, int& height) const {
        const int pixel_size = 4;
        width = kGridSize_h * pixel_size;
        height = kGridSize_v * pixel_size;
        std::vector<uint8_t> image(width * height * 3, 255);  // white background

        auto draw_cell = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
            for (int dy = 0; dy < pixel_size; ++dy) {
                for (int dx = 0; dx < pixel_size; ++dx) {
                    int px = x * pixel_size + dx;
                    int py = y * pixel_size + dy;
                    int idx = (py * width + px) * 3;
                    image[idx + 0] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                }
            }
            };

        for (int y = 0; y < kGridSize_v; ++y) {
            for (int x = 0; x < kGridSize_h; ++x) {
                if (x == agent_x && y == agent_y)
                    draw_cell(x, y, 0, 0, 255);      // Blue for agent
                else if (x == target_x && y == target_y)
                    draw_cell(x, y, 255, 0, 0);      // Red for target
                else
                    draw_cell(x, y, 200, 200, 200);  // Light gray for empty
            }
        }

        return image;
    }

private:
    int agent_x = 0, agent_y = 0;
    int target_x = 0, target_y = 0;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dist_x;
    std::uniform_int_distribution<> dist_y;

    TimeStep time_step;
};

class LossTracker {
private:
    std::vector<float> actor_losses;
    std::vector<float> critic_losses;
    int graph_width = 50;

    void print_graph(const std::vector<float>& losses, const std::string& name) {
        if (losses.empty()) return;

        float max_loss = *std::max_element(losses.begin(), losses.end());
        float min_loss = *std::min_element(losses.begin(), losses.end());

        std::cout << name << " loss graph:\n";

        for (float loss : losses) {
            int bar_len = (int)((loss - min_loss) / (max_loss - min_loss + 1e-8) * graph_width);
            std::cout << "|";
            for (int i = 0; i < bar_len; i++) std::cout << "=";
            std::cout << " " << loss << "\n";
        }
        std::cout << std::endl;
    }

public:
    void add_losses(float actor_loss, float critic_loss) {
        actor_losses.push_back(actor_loss);
        critic_losses.push_back(critic_loss);
    }

    void print_all() {
        print_graph(actor_losses, "Actor");
        print_graph(critic_losses, "Critic");
    }

    void clear() {
        actor_losses.clear();
        critic_losses.clear();
    }
};

struct NormalDistribution {
    torch::Tensor mean;
    torch::Tensor sigma; //Standard deviation

    NormalDistribution(const torch::Tensor& mean, const torch::Tensor& sigma)
        : mean(mean), sigma(sigma) {
    }

    torch::Tensor sample() const {
        return mean + sigma * torch::randn_like(mean);
    }

    torch::Tensor log_prob(const torch::Tensor& value) const {
        static const double log_sqrt_2pi = std::log(std::sqrt(2.0 * M_PI));
        torch::Tensor var = sigma * sigma;
        return -((value - mean).pow(2) / (2.0 * var)) - sigma.log() - log_sqrt_2pi;
    }

    torch::Tensor entropy() const {
        static const double log_sqrt_2pi_e = 0.5 * std::log(2.0 * M_PI * std::exp(1.0));
        return sigma.log() + log_sqrt_2pi_e;
    }
};

struct ObsStats {
    torch::Tensor mean;
    torch::Tensor var;
    int64_t count;

    ObsStats() : mean(torch::zeros({})), var(torch::ones({})), count(0) {}

    void save(torch::serialize::OutputArchive& archive, const std::string& prefix) const {
        archive.write(prefix + "_mean", mean);
        archive.write(prefix + "_var", var);
        archive.write(prefix + "_count", torch::tensor(count, torch::kInt64));
    }

    void load(torch::serialize::InputArchive& archive, const std::string& prefix) {
        archive.read(prefix + "_mean", mean);
        archive.read(prefix + "_var", var);
        torch::Tensor count_tensor;
        archive.read(prefix + "_count", count_tensor);
        count = count_tensor.item<int64_t>();
    }
};

class SimpleTensorBoardLogger {
    std::ofstream file;
    std::mutex mtx;

public:
    SimpleTensorBoardLogger(const std::string& filepath) {
        namespace fs = std::filesystem;
        if (auto dir = fs::path(filepath).parent_path(); !dir.empty())
            fs::create_directories(dir);

        file.open(filepath, std::ios::app);
        if (!file) throw std::runtime_error("Failed to open " + filepath);

        if (file.tellp() == 0) file << "step,tag,value\n";
    }

    void add_scalar(const std::string& tag, double value, int step) {
        std::lock_guard lock(mtx);
        file << step << ',' << tag << ',' << value << '\n';
        file.flush();
    }

    ~SimpleTensorBoardLogger() { if (file.is_open()) file.close(); }
};

void init_weights(torch::nn::Module& module, double gain) {
    if (auto* linear = module.as<torch::nn::Linear>()) {
        torch::nn::init::orthogonal_(linear->weight, gain);
        if (linear->bias.defined()) {
            linear->bias.data().fill_(0.0);
        }
    }
}

// Actor Network
struct ActorNetworkImpl : torch::nn::Module{
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, mu{ nullptr };
    torch::Tensor log_sigma;

    ActorNetworkImpl(int64_t obs_size, int64_t action_size) {
        fc1 = register_module("fc1", torch::nn::Linear(obs_size, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        mu = register_module("mu", torch::nn::Linear(64, action_size));
        log_sigma = register_parameter("log_sigma", torch::zeros({ 1, action_size }));

        std::map<torch::nn::Module*, double> module_gains{
            {fc1.get(), std::sqrt(2.0)},
            {fc2.get(), std::sqrt(2.0)},
            {mu.get(), 0.01}
        };

        for (auto& mg : module_gains) {
            init_weights(*mg.first, mg.second);
        }
    }

    NormalDistribution forward(torch::Tensor x) {
        if (x.dim() == 1) x = x.unsqueeze(0);

        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        auto mu_out = mu->forward(x);
        auto sigma = torch::exp(log_sigma).expand_as(mu_out).to(x.device());
        return NormalDistribution(mu_out, sigma);
    }
};
TORCH_MODULE(ActorNetwork);

// Critic Network
struct CriticNetworkImpl : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, v{ nullptr };

    CriticNetworkImpl(int64_t obs_size, int64_t action_size) {
        fc1 = register_module("fc1", torch::nn::Linear(obs_size, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        v = register_module("v", torch::nn::Linear(64, 1));

        std::map<torch::nn::Module*, double> module_gains{
            {fc1.get(), std::sqrt(2.0)},
            {fc2.get(), std::sqrt(2.0)},
            {v.get(), 1.0}
        };

        for (auto& mg : module_gains) {
            init_weights(*mg.first, mg.second);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        if (x.dim() == 1) x = x.unsqueeze(0);
        x = x.to(torch::kFloat32);

        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        return v->forward(x).view(-1);
    }
};
TORCH_MODULE(CriticNetwork);

// Parses the observation dictionary from the TimeStep
std::tuple<torch::Tensor, double, bool> process_observation(const TimeStep& time_step) {
    std::vector<torch::Tensor> obs_tensors;
    for (const auto& [_, val] : time_step.observation) {
        obs_tensors.push_back(val.unsqueeze(0).to(torch::kFloat32));
    }
    torch::Tensor obs = torch::cat(obs_tensors, 0);
    double r = time_step.reward;
    bool done = time_step.done;
    return { obs, r, done };
}


class PPO {
public:
    PPO(const ArgList& arglist) : arglist(arglist), device(torch::kCPU) {
        // Seed RNGs
        std::srand(arglist.seed);
        torch::manual_seed(arglist.seed);

        device = !torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

        actor = ActorNetwork(env.kObsSize, env.kNumActions);
        actor->to(device);

        if (arglist.mode == "train") {
            critic = CriticNetwork(env.kObsSize, env.kNumActions);
            critic->to(device);

            std::filesystem::path path = "./log/" + arglist.domain + "_" + arglist.task;
            exp_dir = path / ("seed_" + std::to_string(arglist.seed));
            model_dir = exp_dir / "models";
            tensorboard_dir = exp_dir / "tensorboard";

            if (arglist.resume) {
                torch::serialize::InputArchive archive;
                archive.load_from((model_dir / "backup.ckpt").string());

                // Load episode
                torch::Tensor episode_tensor;
                archive.read("episode", episode_tensor);
                start_episode = episode_tensor.item<int>() + 1;

                // Load actor
                torch::serialize::InputArchive actor_archive;
                archive.read("actor", actor_archive);
                actor->load(actor_archive);

                // Load critic
                torch::serialize::InputArchive critic_archive;
                archive.read("critic", critic_archive);
                critic->load(critic_archive);

                // Load obs_stats fields
                archive.read("obs_stats_mean", obs_stats.mean);
                archive.read("obs_stats_var", obs_stats.var);

                torch::Tensor count_tensor;
                archive.read("obs_stats_count", count_tensor);
                obs_stats.count = count_tensor.item<int>();

                // Re-create optimizers
                actor_optimizer = std::make_unique<torch::optim::Adam>(actor->parameters(), torch::optim::AdamOptions(arglist.lr));
                critic_optimizer = std::make_unique<torch::optim::Adam>(critic->parameters(), torch::optim::AdamOptions(arglist.lr));

                // Load optimizer states
                torch::serialize::InputArchive actor_opt_archive;
                archive.read("actor_optim", actor_opt_archive);
                actor_optimizer->load(actor_opt_archive);

                torch::serialize::InputArchive critic_opt_archive;
                archive.read("critic_optim", critic_opt_archive);
                critic_optimizer->load(critic_opt_archive);

                std::cout << "Done loading checkpoint ..." << std::endl;
            }
            else {
              start_episode = 0;

              if (!std::filesystem::exists(path))
                  std::filesystem::create_directories(path);
              std::filesystem::create_directories(exp_dir);
              std::filesystem::create_directories(tensorboard_dir);
              std::filesystem::create_directories(model_dir);

              actor_optimizer = std::make_unique<torch::optim::Adam>(actor->parameters(), torch::optim::AdamOptions(arglist.lr));
              critic_optimizer = std::make_unique<torch::optim::Adam>(critic->parameters(), torch::optim::AdamOptions(arglist.lr));
            }
            train_simplified();
        }
    }

    torch::Tensor normalize_observation(const torch::Tensor& o, bool update = false) {
        static torch::Tensor running_mean = torch::zeros_like(o);
        static torch::Tensor running_var = torch::ones_like(o);
        static int64_t count = 1;

        if (update) {
            count += 1;
            torch::Tensor delta = o - running_mean;
            running_mean += delta / count;
            torch::Tensor delta2 = o - running_mean;
            running_var += delta * delta2;
        }

        torch::Tensor var_s = running_var / count;
        torch::Tensor mean = running_mean;

        bool has_nan = var_s.isnan().any().item<bool>();
        bool has_zero = (var_s == 0).any().item<bool>();

        torch::Tensor norm_obs;
        if (has_nan || has_zero) {
            norm_obs = o - mean;
        }
        else {
            norm_obs = (o - mean) / torch::sqrt(var_s + 1e-8);
        }

        return torch::clamp(norm_obs, -10.0, 10.0);
    }

    // Save checkpoint function
    void save_checkpoint(const std::string& name) {
        try {
            torch::serialize::OutputArchive archive;

            torch::serialize::OutputArchive actor_archive;
            actor->save(actor_archive);
            archive.write("actor", actor_archive);

            torch::serialize::OutputArchive critic_archive;
            critic->save(critic_archive);
            archive.write("critic", critic_archive);

            archive.write("obs_stats_mean", obs_stats.mean);
            archive.write("obs_stats_var", obs_stats.var);
            archive.write("obs_stats_count", torch::tensor(obs_stats.count, torch::kInt32));
            std::string save_file = (model_dir / name).string();
            archive.save_to(save_file);
        }
        catch (const c10::Error& e) {
            std::cerr << "Exception during save_checkpoint: " << e.what() << std::endl;
        }
    }

    void save_backup(int episode) {
        try {
            torch::serialize::OutputArchive archive;
            archive.write("episode", torch::tensor({ episode }, torch::kInt32));
            torch::serialize::OutputArchive actor_archive;
            actor->save(actor_archive);
            archive.write("actor", actor_archive);
            torch::serialize::OutputArchive critic_archive;
            critic->save(critic_archive);
            archive.write("critic", critic_archive);
            torch::serialize::OutputArchive actor_optim_archive;
            actor_optimizer->save(actor_optim_archive);
            archive.write("actor_optim", actor_optim_archive);
            torch::serialize::OutputArchive critic_optim_archive;
            critic_optimizer->save(critic_optim_archive);
            archive.write("critic_optim", critic_optim_archive);
            archive.write("obs_stats_mean", obs_stats.mean);
            archive.write("obs_stats_var", obs_stats.var);
            archive.write("obs_stats_count", torch::tensor(obs_stats.count, torch::kInt32));
            archive.save_to((model_dir / "backup.ckpt").string());
        }
        catch (const c10::Error& e) {
            std::cerr << "Exception during save_backup: " << e.what() << std::endl;
        }
    }

    void train_simplified() {
        SimpleTensorBoardLogger logger("runs/log_dir");

        struct RolloutResult {
            std::vector<torch::Tensor> observations;
            std::vector<torch::Tensor> actions;
            std::vector<torch::Tensor> old_log_probs;
            std::vector<torch::Tensor> state_values;
            std::vector<torch::Tensor> rewards;
            torch::Tensor current_observation;
            double total_reward;
            bool done;
        };

        auto collect_rollout = [&](torch::Tensor start_obs) -> RolloutResult {
            RolloutResult result;
            result.current_observation = start_obs;
            result.total_reward = 0.0;
            int timestep = 0;
            while (true) {
                torch::Tensor obs_batch = result.current_observation.unsqueeze(0).to(device);
                auto dist = actor->forward(obs_batch);
                auto value = critic->forward(obs_batch)[0];
                auto action = dist.sample();
                result.observations.push_back(obs_batch[0]);
                result.actions.push_back(action[0].detach());
                result.old_log_probs.push_back(dist.log_prob(action).sum().detach());
                result.state_values.push_back(value);
                auto [next_raw_obs, reward, done] = process_observation(env.step(action));
                result.current_observation = normalize_observation(next_raw_obs, true);
                result.rewards.push_back(torch::tensor(reward, torch::kFloat32).to(device));
                result.total_reward += reward;
                timestep++;
                if (done || (timestep % arglist.update_every) == 0) {
                    result.done = done;
                    result.state_values.push_back(critic->forward(result.current_observation.unsqueeze(0).to(device))[0].detach());
                    break;
                }
            }
            return result;
            };

        auto compute_returns_and_advantages = [&](const RolloutResult& data) {
            std::vector<torch::Tensor> returns;
            torch::Tensor discounted = data.state_values.back();
            for (int i = static_cast<int>(data.rewards.size()) - 1; i >= 0; --i) {
                discounted = data.rewards[i] + arglist.gamma * discounted;
                returns.push_back(discounted);
            }
            std::reverse(returns.begin(), returns.end());
            auto obs_tensor = torch::stack(data.observations);
            auto act_tensor = torch::stack(data.actions);
            auto old_log_tensor = torch::stack(data.old_log_probs);
            auto value_tensor = torch::stack(data.state_values).slice(0, 0, returns.size());
            auto ret_tensor = torch::stack(returns);
            auto adv_tensor = (ret_tensor - value_tensor).detach();

            // Print advantage stats early
            std::cout << "Advantage mean: " << adv_tensor.mean().item<double>()
                << ", std: " << adv_tensor.std().item<double>() << std::endl;

            return std::make_tuple(obs_tensor, act_tensor, old_log_tensor, value_tensor, ret_tensor, adv_tensor);
            };

        auto run_ppo_update = [&](torch::Tensor obs, torch::Tensor act, torch::Tensor old_log, torch::Tensor val,
            torch::Tensor rets, torch::Tensor adv, int episode_index) {
                for (int epoch = 0; epoch < arglist.epochs; ++epoch) {
                    auto perm = torch::randperm(obs.size(0));
                    double actor_loss_sum = 0.0, critic_loss_sum = 0.0;
                    int batch_count = 0;
                    for (int start = 0; start < obs.size(0); start += arglist.batch_size) {
                        auto indices = perm.slice(0, start, std::min(start + arglist.batch_size, (int)obs.size(0)));
                        auto b_obs = obs.index_select(0, indices);
                        auto b_act = act.index_select(0, indices);
                        auto b_old_log = old_log.index_select(0, indices);
                        auto b_adv = adv.index_select(0, indices);
                        auto b_ret = rets.index_select(0, indices);
                        if (b_adv.numel() > 1)
                            b_adv = (b_adv - b_adv.mean()) / torch::clamp(b_adv.std(), 1e-5);

                        auto dist = actor->forward(b_obs);
                        auto val_est = critic->forward(b_obs);
                        auto log_probs = dist.log_prob(b_act).sum(1);
                        auto ratio = torch::exp(log_probs - b_old_log);

                        // Print policy ratio stats before clipping
                        std::cout << "Ratio mean: " << ratio.mean().item<double>()
                            << ", std: " << ratio.std().item<double>() << std::endl;

                        auto surr1 = ratio * b_adv;
                        auto surr2 = torch::clamp(ratio, 1 - arglist.ppo_clip_term, 1 + arglist.ppo_clip_term) * b_adv;
                        auto actor_loss = -torch::min(surr1, surr2).mean();
                        auto critic_loss = 0.5 * torch::mse_loss(val_est, b_ret);

                        // Print losses per batch
                        std::cout << "Actor loss: " << actor_loss.item<double>()
                            << ", Critic loss: " << critic_loss.item<double>() << std::endl;

                        if (actor_loss.isnan().item<bool>() || critic_loss.isnan().item<bool>()) {
                            std::cerr << "NaN detected in losses. Skipping batch update.\n";
                            break;
                        }
                        actor_loss_sum += actor_loss.item<double>();
                        critic_loss_sum += critic_loss.item<double>();
                        batch_count++;
                        auto total_loss = actor_loss + critic_loss;
                        try {
                            actor_optimizer->zero_grad();
                            critic_optimizer->zero_grad();
                            total_loss.backward();
                            torch::nn::utils::clip_grad_norm_(actor->parameters(), arglist.gradient_clip_term);
                            torch::nn::utils::clip_grad_norm_(critic->parameters(), arglist.gradient_clip_term);
                            actor_optimizer->step();
                            critic_optimizer->step();
                        }
                        catch (const c10::Error& e) {
                            std::cerr << "Backward pass error: " << e.what() << "\n";
                        }
                    }
                    if (batch_count > 0) {
                        std::cout << "Episode " << episode_index << " Epoch " << epoch
                            << " Actor Loss: " << actor_loss_sum / batch_count
                            << " Critic Loss: " << critic_loss_sum / batch_count << std::endl;
                    }
                }
            };

        for (int episode_index = start_episode; episode_index < arglist.episodes; ++episode_index) {
            auto [raw_obs, _, __] = process_observation(env.reset());
            torch::Tensor obs = normalize_observation(raw_obs, true);
            auto rollout = collect_rollout(obs);
            auto [obs_tensor, act_tensor, old_log_tensor, val_tensor, ret_tensor, adv_tensor] = compute_returns_and_advantages(rollout);
            run_ppo_update(obs_tensor, act_tensor, old_log_tensor, val_tensor, ret_tensor, adv_tensor, episode_index);
            if (rollout.done) {
                std::cout << "Episode " << episode_index << " average reward: " << rollout.total_reward << std::endl;
                logger.add_scalar("episode_reward", rollout.total_reward, episode_index);
                if (episode_index % arglist.eval_every == 0 || episode_index == arglist.episodes - 1) {
                    auto eval_rewards = eval(arglist.eval_over, true, true);
                    double mean = std::accumulate(eval_rewards.begin(), eval_rewards.end(), 0.0) / eval_rewards.size();
                    logger.add_scalar("evaluation_reward", mean, episode_index);
                    save_checkpoint(std::to_string(episode_index) + ".ckpt");
                }
                if ((episode_index % 250 == 0 || episode_index == arglist.episodes - 1) && episode_index > start_episode) {
                    save_backup(episode_index);
                }
            }
        }
    }



    std::vector<double> eval(int episodes, bool render = false, bool save_video = false) {
        int t = 0;
        std::string folder;
        if (render && save_video) {
            folder = "./media/" + arglist.domain + "_" + arglist.task;
            std::system(("mkdir -p " + folder).c_str());
        }

        std::vector<double> ep_r_list;

        for (int episode = 0; episode < episodes; ++episode) {
            torch::Tensor o;
            double ep_r = 0.0;
            decltype(env.reset()) reset_result = env.reset();
            std::tie(o, std::ignore, std::ignore) = process_observation(reset_result);
            o = normalize_observation(o);

            //matplotlibcpp::image* vid = nullptr;

            while (true) {
                torch::NoGradGuard no_grad;
                auto dist = actor->forward(o.to(torch::kFloat32).unsqueeze(0).to(device));
                auto a_tensor = dist.sample();

                //std::cout << "dist.mean: " << dist.mean << std::endl;
                //std::cout << "dist.stddev: " << dist.sigma << std::endl;
                //std::cout << "dist.entropy: " << dist.entropy() << std::endl;
                //std::cout << "a_tensor: " << a_tensor << std::endl;
                //auto a = a_tensor.cpu().squeeze(0).item<double>();

                auto step_result = env.step(a_tensor);
                torch::Tensor o_1;
                double r;
                bool done;
                std::tie(o_1, r, done) = process_observation(step_result);
                o_1 = normalize_observation(o_1);

                if (render) {
                    try {
                        int width, height;
                        auto img = env.render(width, height); // returns std::vector<uint8_t> RGB image

                        if (save_video) {
                            if (!ffmpeg_started) {
                                std::filesystem::create_directories(folder);
                                std::string output_path = folder + "/output.mp4";
                                ffmpeg_writer = std::make_unique<FFmpegWriter>(output_path, width, height, fps);
                                ffmpeg_started = true;
                            }

                            ffmpeg_writer->write_frame(img.data());
                        }
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Render or video write failed: " << e.what() << std::endl;
                    }
                }

                ep_r += r;
                o = o_1;

                if (done) {
                    ep_r_list.push_back(ep_r);
                    if (render) {
                        std::cout << "Episode finished with total reward " << ep_r << std::endl;
                        //matplotlibcpp::pause(0.5);
                    }
                    break;
                }
            }
        }

        if (arglist.mode == "eval") {
            double mean_return = std::accumulate(ep_r_list.begin(), ep_r_list.end(), 0.0) / ep_r_list.size();
            std::cout << "Average return : " << mean_return << std::endl;

            if (save_video) {
                std::system(("cd " + folder + " && ffmpeg -i file%04d.png -r 10 -vf pad=ceil(iw/2)*2:ceil(ih/2)*2 -pix_fmt yuv420p video.mp4").c_str());
                std::system(("cd " + folder + " && rm *.png").c_str());
                std::system(("cd " + folder + " && ffmpeg -i video.mp4 video.gif").c_str());
            }
        }

        return ep_r_list;
    }

private:
    ArgList arglist;
    GridWorldEnv env;
    torch::Device device;
    int start_episode;
    ActorNetwork actor = nullptr;
    CriticNetwork critic = nullptr;

    std::filesystem::path exp_dir, model_dir, tensorboard_dir;

    std::unique_ptr<torch::optim::Adam> actor_optimizer;
    std::unique_ptr<torch::optim::Adam> critic_optimizer;
    ObsStats obs_stats;
    LossTracker loss_tracker;

    int frame_width = 640;
    int frame_height = 480;
    int fps = 30;
    bool ffmpeg_started = false;
    std::unique_ptr<FFmpegWriter> ffmpeg_writer;
};

int main() {

    write_rgb_pattern_mp4();

    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available. GPU will be used.\n";
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);

        if (err != cudaSuccess || deviceCount == 0) {
            std::cout << "Failed to get CUDA device count or no devices found.\n";
            return 1;
        }

        std::cout << "CUDA device count: " << deviceCount << "\n";

        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, 0);
        if (err == cudaSuccess) {
            std::cout << "Current device name: " << deviceProp.name << "\n";
        }
        else {
            std::cout << "Failed to get device properties.\n";
        }
    }
    else {
        std::cout << "CUDA is NOT available. CPU will be used.\n";
    }

    ArgList arglist;
    PPO ppo(arglist);

    return 0;
}