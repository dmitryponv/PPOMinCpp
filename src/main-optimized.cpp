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
#include <chrono>
#include <unordered_map>

using namespace std;

void print_tensor_inline(const std::string& name, const torch::Tensor& t, int precision = 4, int max_elements = 10) {
    //torch::Tensor flat = t.flatten().cpu();
    //std::cout << name << "=tensor([";
    //int64_t size = flat.size(0);
    //std::cout << std::fixed << std::setprecision(precision);
    //for (int64_t i = 0; i < std::min<int64_t>(size, max_elements / 2); ++i) {
    //   std::cout << flat[i].item<double>() << ", ";
    //}
    //if (size > max_elements) {
    //   std::cout << "...";
    //   for (int64_t i = size - max_elements / 2; i < size; ++i) {
    //       std::cout << ", " << flat[i].item<double>();
    //   }
    //}
    //std::cout << "])" << std::endl << std::endl;
}

//Abstract environment interface
class Env {
public:
    struct Space {
        vector<int> shape;
    };

    virtual ~Env() = default;

    //Reset the environment and return initial observation
    virtual pair<torch::Tensor, unordered_map<string, float>> reset() = 0;

    //Step the environment with given action returns: observation, reward, terminated flag, truncated flag, extra info
    virtual tuple<torch::Tensor, float, bool, bool, unordered_map<string, float>> step(const torch::Tensor& action) = 0;

    virtual void render() = 0;

    virtual Space observation_space() const = 0;

    virtual Space action_space() const = 0;
};

class CartPoleEnv : public Env {
private:
    float gravity = 9.8f;
    float masscart = 1.0f;
    float masspole = 0.1f;
    float total_mass = masspole + masscart;
    float length = 0.5f;  //half the pole's length
    float polemass_length = masspole * length;
    float force_mag = 10.0f;
    float tau = 0.02f;
    float theta_threshold_radians = 12 * 2 * M_PI / 360;
    float x_threshold = 2.4f;

    vector<float> state;
    int steps_beyond_terminated = -1;
    string integrator = "euler";

    mt19937 rng;
    uniform_real_distribution<float> dist{ -0.05f, 0.05f };

    torch::Device& mDevice;

public:
    CartPoleEnv(torch::Device& device): mDevice(device) {
        random_device rd;
        rng = mt19937(rd());
    }

    Space observation_space() const override {
        return Space{ {4} };
    }

    Space action_space() const override {
        return Space{ {1} };  //Discrete(2)
    }

    pair<torch::Tensor, unordered_map<string, float>> reset() override {
        state = { dist(rng), dist(rng), dist(rng), dist(rng) };
        steps_beyond_terminated = -1;
        torch::Tensor state_tensor = torch::tensor(state).to(mDevice);
        return { state_tensor, {} };
    }

    tuple<torch::Tensor, float, bool, bool, unordered_map<string, float>> step(const torch::Tensor& action) override {
        assert(action.numel() == 1);
        float action_value = action.item<float>();

        float x = state[0];
        float x_dot = state[1];
        float theta = state[2];
        float theta_dot = state[3];

        float force = (action_value > 0) ? force_mag : -force_mag;

        float costheta = cos(theta);
        float sintheta = sin(theta);

        float temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        float thetaacc = (gravity * sintheta - costheta * temp) /
            (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass));
        float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        if (integrator == "euler") {
            x += tau * x_dot;
            x_dot += tau * xacc;
            theta += tau * theta_dot;
            theta_dot += tau * thetaacc;
        }
        else {
            x_dot += tau * xacc;
            x += tau * x_dot;
            theta_dot += tau * thetaacc;
            theta += tau * theta_dot;
        }

        state = { x, x_dot, theta, theta_dot };

        bool terminated = (x < -x_threshold || x > x_threshold ||
            theta < -theta_threshold_radians || theta > theta_threshold_radians);

        float reward = 0.0f;
        if (!terminated) {
            reward = 1.0f;
        }
        else if (steps_beyond_terminated == -1) {
            steps_beyond_terminated = 0;
            reward = 1.0f;
        }
        else {
            if (steps_beyond_terminated == 0) {
                printf("Warning: step() called after environment is terminated. Call reset().\n");
            }
            steps_beyond_terminated += 1;
            reward = 0.0f;
        }

        torch::Tensor next_state = torch::tensor({ x, x_dot, theta, theta_dot }).to(mDevice);
        return { next_state, reward, terminated, false, {} };
    }

    void render() override {
        //Rendering can be implemented here using SDL2, SFML, OpenGL, or skipped.
        printf("State: [%.3f, %.3f, %.3f, %.3f]\n", state[0], state[1], state[2], state[3]);
    }
};

class PendulumEnv : public Env {
public:
    PendulumEnv(torch::Device& device, const string& render_mode = "", float gravity = 10.0f)
        : mDevice(device), g(gravity), render_mode(render_mode) {
        max_speed = 8.0f;
        max_torque = 2.0f;
        dt = 0.05f;
        m = 1.0f;
        l = 1.0f;

        obs_space.shape = { 3 };
        act_space.shape = { 1 };

        random_device rd;
        rng = mt19937(rd());
        dist = uniform_real_distribution<float>(-3.14f, 3.14f);
    }

    pair<torch::Tensor, unordered_map<string, float>> reset() override {
        float theta = dist(rng);
        float theta_dot = dist(rng) / 4.0f;
        state = { theta, theta_dot };
        last_u = 0.0f;
        return { get_obs(), {} };
    }

    tuple<torch::Tensor, float, bool, bool, unordered_map<string, float>> step(const torch::Tensor& action) override {
        float u = std::clamp(action.item<float>(), -max_torque, max_torque);
        last_u = u;

        float theta = state[0];
        float theta_dot = state[1];

        float cost = angle_normalize(theta) * angle_normalize(theta)
            + 0.1f * theta_dot * theta_dot
            + 0.001f * u * u;

        float new_theta_dot = theta_dot + (3.0f * g / (2.0f * l) * std::sin(theta) + 3.0f / (m * l * l) * u) * dt;
        new_theta_dot = std::clamp(new_theta_dot, -max_speed, max_speed);
        float new_theta = theta + new_theta_dot * dt;

        state = { new_theta, new_theta_dot };

        return { get_obs(), -cost, false, false, {} };
    }

    void render() override {
        if (render_mode == "human") {
            cout << "Angle: " << state[0] << ", Angular velocity: " << state[1] << ", Torque: " << last_u << "\n";
        }
    }

    Space observation_space() const override {
        return obs_space;
    }

    Space action_space() const override {
        return act_space;
    }

private:
    torch::Device& mDevice;
    float g, m, l, dt;
    float max_speed, max_torque;
    float last_u;
    string render_mode;
    vector<float> state;
    Space obs_space, act_space;

    mt19937 rng;
    uniform_real_distribution<float> dist;

    torch::Tensor get_obs() const {
        float theta = state[0];
        float theta_dot = state[1];
        return torch::tensor({ std::cos(theta), std::sin(theta), theta_dot }).to(mDevice);
    }

    float angle_normalize(float x) const {
        return fmodf(x + M_PI, 2.0f * M_PI) - M_PI;
    }

    float clamp(float v, float lo, float hi) const {
        return std::max(lo, std::min(v, hi));
    }
};

class AgentTargetEnv : public Env {
private:
    float x_min = 0.0f, x_max = 10.0f;
    float y_min = 0.0f, y_max = 10.0f;
    float max_step = 0.5f;  //max movement per step in each axis

    vector<float> agent_pos;  //{x, y}
    vector<float> target_pos; //{x, y}

    mt19937 rng;
    uniform_real_distribution<float> dist_x;
    uniform_real_distribution<float> dist_y;

public:
    AgentTargetEnv(torch::Device& device)
        : mDevice(device), dist_x(x_min, x_max), dist_y(y_min, y_max) {
        random_device rd;
        rng = mt19937(rd());
    }

    Space observation_space() const override {
        //Observations: agent_x, agent_y, target_x, target_y
        return Space{ {4} };
    }

    Space action_space() const override {
        //Actions: continuous 2 floats, each in [-1, 1]
        return Space{ {2} };
    }

    pair<torch::Tensor, unordered_map<string, float>> reset() override {
        agent_pos = { dist_x(rng), dist_y(rng) };
        target_pos = { dist_x(rng), dist_y(rng) };
        return { get_observation(), {} };
    }

    tuple<torch::Tensor, float, bool, bool, unordered_map<string, float>> step(const torch::Tensor& action) override {
        //Clip action to [-1, 1]
        float dx = std::clamp(action[0].item<float>(), -1.0f, 1.0f) * max_step;
        float dy = std::clamp(action[1].item<float>(), -1.0f, 1.0f) * max_step;

        //Update agent position and clip to bounds
        agent_pos[0] = std::clamp(agent_pos[0] + dx, x_min, x_max);
        agent_pos[1] = std::clamp(agent_pos[1] + dy, y_min, y_max);

        //Distance to target
        float dist_x = agent_pos[0] - target_pos[0];
        float dist_y = agent_pos[1] - target_pos[1];
        float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);

        //Reward and done conditions
        float reward = -0.01f * distance;
        bool done = false;

        if (distance < 1.0f) {
            reward += 1.0f;
            done = true;
        }

        return { get_observation(), reward, done, false, {} };
    }

    void render() override {
        printf("Agent: (%.2f, %.2f), Target: (%.2f, %.2f)\n", agent_pos[0], agent_pos[1], target_pos[0], target_pos[1]);
    }

private:
    torch::Device& mDevice;

    torch::Tensor get_observation() const {
        return torch::tensor({ agent_pos[0], agent_pos[1], target_pos[0], target_pos[1] }).to(mDevice);
    }
};
///////////////////////////////////////////////////////////////////////////////////////////////////////

class MultivariateNormal {
public:
    torch::Tensor loc;
    torch::Tensor _unbroadcasted_scale_tril;
    torch::Tensor scale_tril_;
    torch::Tensor covariance_matrix_;
    torch::Tensor precision_matrix_;
    std::vector<int64_t> batch_shape;
    std::vector<int64_t> event_shape;

    MultivariateNormal(const torch::Tensor& loc,
        const torch::optional<torch::Tensor>& covariance_matrix = torch::nullopt,
        const torch::optional<torch::Tensor>& precision_matrix = torch::nullopt,
        const torch::optional<torch::Tensor>& scale_tril = torch::nullopt,
        torch::Device device = torch::kCPU) {
        if (loc.dim() < 1) {
            throw std::invalid_argument("loc must be at least one-dimensional.");
        }

        int specified = (bool)covariance_matrix + (bool)precision_matrix + (bool)scale_tril;
        if (specified != 1) {
            throw std::invalid_argument("Exactly one of covariance_matrix, precision_matrix, or scale_tril must be specified.");
        }

        if (scale_tril.has_value()) {
            if (scale_tril->dim() < 2) {
                throw std::invalid_argument("scale_tril must be at least two-dimensional.");
            }
            auto bshape = broadcast_shapes(scale_tril->sizes().vec(), loc.sizes().vec(), 2, 1);
            scale_tril_ = scale_tril->expand(bshape).contiguous();
            _unbroadcasted_scale_tril = *scale_tril;
            batch_shape = std::vector<int64_t>(bshape.begin(), bshape.end() - 2);
        }
        else if (covariance_matrix.has_value()) {
            if (covariance_matrix->dim() < 2) {
                throw std::invalid_argument("covariance_matrix must be at least two-dimensional.");
            }
            auto bshape = broadcast_shapes(covariance_matrix->sizes().vec(), loc.sizes().vec(), 2, 1);
            covariance_matrix_ = covariance_matrix->expand(bshape).contiguous();
            _unbroadcasted_scale_tril = torch::linalg_cholesky(*covariance_matrix);
            batch_shape = std::vector<int64_t>(bshape.begin(), bshape.end() - 2);
        }
        else {
            if (precision_matrix->dim() < 2) {
                throw std::invalid_argument("precision_matrix must be at least two-dimensional.");
            }
            auto bshape = broadcast_shapes(precision_matrix->sizes().vec(), loc.sizes().vec(), 2, 1);
            precision_matrix_ = precision_matrix->expand(bshape).contiguous();
            _unbroadcasted_scale_tril = torch::linalg_cholesky(torch::linalg_inv(*precision_matrix));
            batch_shape = std::vector<int64_t>(bshape.begin(), bshape.end() - 2);
        }

        auto expanded_shape = batch_shape;
        expanded_shape.push_back(loc.size(-1));
        this->loc = loc.expand(expanded_shape).contiguous();
        event_shape = { loc.size(-1) };
    }

    torch::Tensor scale_tril() const {
        auto shape = batch_shape;
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        return _unbroadcasted_scale_tril.expand(shape);
    }

    torch::Tensor covariance_matrix() const {
        auto L = _unbroadcasted_scale_tril;
        auto shape = batch_shape;
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        return torch::matmul(L, L.transpose(-2, -1)).expand(shape);
    }

    torch::Tensor precision_matrix() const {
        auto shape = batch_shape;
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        return torch::cholesky_inverse(_unbroadcasted_scale_tril).expand(shape);
    }

    torch::Tensor mean() const {
        return loc;
    }

    torch::Tensor mode() const {
        return loc;
    }

    torch::Tensor variance() const {
        auto shape = batch_shape;
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        return _unbroadcasted_scale_tril.pow(2).sum(-1).expand(shape);
    }

    torch::Tensor sample(const std::vector<int64_t>& sample_shape = {}) const {
        torch::NoGradGuard no_grad;
        auto shape = sample_shape;
        shape.insert(shape.end(), loc.sizes().begin(), loc.sizes().end());
        auto eps = torch::randn(shape, loc.options());
        auto L = _unbroadcasted_scale_tril;
        auto result = loc + torch::matmul(L, eps.unsqueeze(-1)).squeeze(-1);
        return result;
    }

    torch::Tensor log_prob(const torch::Tensor& value) const {
        auto diff = value.to(loc.device()) - loc;
        auto M = batch_mahalanobis(_unbroadcasted_scale_tril, diff);
        auto half_log_det = _unbroadcasted_scale_tril.diagonal(0, -2, -1).log().sum(-1);
        return -0.5 * (event_shape[0] * std::log(2 * M_PI) + M) - half_log_det;
    }

    torch::Tensor entropy() const {
        auto half_log_det = _unbroadcasted_scale_tril.diagonal(0, -2, -1).log().sum(-1);
        return 0.5 * event_shape[0] * (1.0 + std::log(2 * M_PI)) + half_log_det;
    }

private:
    static torch::Tensor batch_mahalanobis(const torch::Tensor& L, const torch::Tensor& diff) {
        auto solve = torch::linalg_solve_triangular(L, diff.unsqueeze(-1), /*upper=*/false).squeeze(-1);
        return solve.pow(2).sum(-1);
    }

    static std::vector<int64_t> broadcast_shapes(std::vector<int64_t> a, std::vector<int64_t> b, int a_end = 0, int b_end = 0) {
        auto a_prefix = std::vector<int64_t>(a.begin(), a.end() - a_end);
        auto b_prefix = std::vector<int64_t>(b.begin(), b.end() - b_end);
        size_t ndim = std::max(a_prefix.size(), b_prefix.size());
        std::vector<int64_t> result(ndim, 1);
        for (int i = ndim - 1, ai = a_prefix.size() - 1, bi = b_prefix.size() - 1; i >= 0; --i, --ai, --bi) {
            int64_t a_dim = ai >= 0 ? a_prefix[ai] : 1;
            int64_t b_dim = bi >= 0 ? b_prefix[bi] : 1;
            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                throw std::invalid_argument("Incompatible shapes for broadcasting");
            }
            result[i] = std::max(a_dim, b_dim);
        }
        result.insert(result.end(), a.end() - a_end, a.end());
        return result;
    }
};

struct FeedForwardNNImpl : torch::nn::Module {
    torch::nn::Linear layer1{ nullptr }, layer2{ nullptr }, layer3{ nullptr };

    FeedForwardNNImpl(int in_dim, int out_dim, torch::Device& device) {
        
        layer1 = register_module("layer1", torch::nn::Linear(in_dim, 64));
        layer2 = register_module("layer2", torch::nn::Linear(64, 64));
        layer3 = register_module("layer3", torch::nn::Linear(64, out_dim));

        layer1->to(device);
        layer2->to(device);
        layer3->to(device);
    }

    torch::Tensor forward(torch::Tensor obs) {
        try {
            //obs = obs.to(layer1->weight.device());
            auto activation1 = torch::relu(layer1(obs));
            auto activation2 = torch::relu(layer2(activation1));
            return layer3(activation2);
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in forward: " << e.what() << std::endl;
            throw;  //rethrow or handle as needed
        }
    }
};
TORCH_MODULE(FeedForwardNN);

class PPO {
public:

    PPO(Env& env, const std::unordered_map<std::string, float>& hyperparameters, torch::Device& device, string actor_model = "", string critic_model = "")
        : env(env), mDevice(device) {

        if (!actor_model.empty() && !critic_model.empty()) {
            cout << "Loading in " << actor_model << " and " << critic_model << "..." << endl;
            torch::load(actor, actor_model);
            torch::load(critic, critic_model);
            cout << "Successfully loaded." << endl;
        }
        else if (!actor_model.empty() || !critic_model.empty()) {
            cerr << "Error: Actor and Critic models must be Specified or Empty" << endl;
            exit(0);
        }
        else {
            cout << "Training from scratch." << endl;
        }

        _init_hyperparameters(hyperparameters);

        obs_dim = env.observation_space().shape[0];
        act_dim = env.action_space().shape[0];

        actor = FeedForwardNN(obs_dim, act_dim, device);
        critic = FeedForwardNN(obs_dim, 1, device);

        actor_optim = std::make_unique<torch::optim::Adam>(actor->parameters(), torch::optim::AdamOptions(lr));
        critic_optim = std::make_unique<torch::optim::Adam>(critic->parameters(), torch::optim::AdamOptions(lr));

        cov_var = torch::full({ act_dim }, 0.5).to(device);
        cov_mat = cov_var.diag();

        logger["delta_t"] = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        logger["t_so_far"] = 0;
        logger["i_so_far"] = 0;
    }

    void learn(int total_timesteps) {
        std::cout << "Learning... Running " << max_timesteps_per_episode
            << " timesteps per episode, " << timesteps_per_batch
            << " timesteps per batch for a total of "
            << total_timesteps << " timesteps" << std::endl;

        int t_so_far = 0;
        int i_so_far = 0;

        while (t_so_far < total_timesteps) {
            //ALG STEP 2-3: Rollout to collect a batch of trajectories
            auto [batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lengths, batch_vals, batch_dones] = rollout_train();

            torch::Tensor A_k = calculate_gae(batch_rews, batch_vals, batch_dones, gamma, lambda);
            torch::Tensor V = critic->forward(batch_obs).squeeze();
            torch::Tensor batch_rtgs = A_k + V.detach();

            t_so_far += std::accumulate(batch_lengths.begin(), batch_lengths.end(), 0);
            i_so_far += 1;

            //Update learning rate 
            float frac = (t_so_far - 1.0f) / total_timesteps;
            float new_lr = lr * (1.0f - frac);
            new_lr = std::max(new_lr, 0.0f);
            actor_optim->param_groups()[0].options().set_lr(new_lr);
            critic_optim->param_groups()[0].options().set_lr(new_lr);

            //Logging
            logger["t_so_far"] = t_so_far;
            logger["i_so_far"] = i_so_far;


            //Get batch size and minibatch size
            int step = batch_obs.size(0);
            std::vector<int> inds(step);
            std::iota(inds.begin(), inds.end(), 0);
            int minibatch_size = step / num_minibatches;

            ///TEMP
            print_tensor_inline("batch_obs", batch_obs);
            print_tensor_inline("batch_acts", batch_acts);
            print_tensor_inline("batch_log_probs", batch_log_probs);
            print_tensor_inline("batch_rtgs", batch_rtgs);
            print_tensor_inline("V", V);
            print_tensor_inline("A_k", A_k);

            //PPO update for multiple epochs
            float approx_kl = 0;
            for (int epoch = 0; epoch < n_updates_per_iteration; ++epoch) {

                //Shuffle indices
                std::shuffle(inds.begin(), inds.end(), std::mt19937{ std::random_device{}() });

                for (int start = 0; start < step; start += minibatch_size) {
                    int end = std::min(start + minibatch_size, step);
                    std::vector<int> idx(inds.begin() + start, inds.begin() + end);
                    torch::Tensor indices = torch::tensor(idx, torch::kLong).reshape({ -1 }).to(mDevice);

                    //Mini-batch slices
                    auto mini_obs = batch_obs.index_select(0, indices);
                    auto mini_acts = batch_acts.index_select(0, indices);
                    auto mini_log_probs = batch_log_probs.index_select(0, indices);
                    auto mini_advantages = A_k.index_select(0, indices);
                    auto mini_rtgs = batch_rtgs.index_select(0, indices);

                    //Forward pass through model
                    auto [V, curr_log_probs, entropy] = evaluate(mini_obs, mini_acts);

                    //PPO objectives
                    auto logratios = curr_log_probs - mini_log_probs;
                    auto ratios = torch::exp(logratios);
                    approx_kl = ((ratios - 1) - logratios).mean().item<float>();
                    auto surr1 = ratios * mini_advantages;
                    auto surr2 = torch::clamp(ratios, 1 - clip, 1 + clip) * mini_advantages;
                    auto actor_loss = -torch::min(surr1, surr2).mean();
                    auto critic_loss = torch::mse_loss(V, mini_rtgs);

                    //Backpropagate actor loss
                    actor_optim->zero_grad();
                    actor_loss.backward();
                    torch::nn::utils::clip_grad_norm_(actor->parameters(), max_grad_norm);
                    actor_optim->step();

                    critic_optim->zero_grad();
                    critic_loss.backward();
                    torch::nn::utils::clip_grad_norm_(critic->parameters(), max_grad_norm);
                    critic_optim->step();

                    torch::Tensor entropy_loss = entropy.mean();
                    actor_loss = actor_loss - ent_coef * entropy_loss;

                    //Logging actor loss and critic loss
                    logger["actor_loss"] = actor_loss;
                    logger["critic_loss"] = critic_loss;
                }
                if (approx_kl > target_kl)
                    break; //if kl aboves threshold
            }
            //Print training summary
            _log_train();

            //Save model
            if (i_so_far % save_freq == 0) {
                if (!std::filesystem::exists("./models")) {
                    std::filesystem::create_directories("./models");
                }
                std::cout << "Saving training model as /models/ppo_actor.pt" << std::endl;
                torch::save(actor, "./models/ppo_actor.pt");
                torch::save(critic, "./models/ppo_critic.pt");
            }
        }
    }

private:
    void _init_hyperparameters(const unordered_map<string, float>& hyperparameters) {
        //Initialize default values for hyperparameters
        timesteps_per_batch = 4800;               //Number of timesteps to run per batch
        max_timesteps_per_episode = 1600;         //Max number of timesteps per episode
        n_updates_per_iteration = 5;              //Number of times to update actor/critic per iteration
        lr = 0.005;                              //Learning rate of actor optimizer
        gamma = 0.95;                            //Discount factor to be applied when calculating GAE
        lambda = 0.98;                            //Discount factor to be applied when calculating GAE
        clip = 0.2;                              //Recommended 0.2, helps define the threshold to clip the ratio during SGA

        //Miscellaneous parameters
        render = false;                          //If we should render during rollout
        render_every_i = 10;                    //Only render every n iterations
        save_freq = 10;                         //How often we save in number of iterations
        seed = nullopt;                    //Sets the seed of our program, used for reproducibility of results
        num_minibatches = 6;
        ent_coef = 0.1f;
        max_grad_norm = 0.5f;
        target_kl = 0.02f;
        //Change any default values to custom values for specified hyperparameters
        for (const auto& [param, val] : hyperparameters) {
            if (param == "timesteps_per_batch") timesteps_per_batch = static_cast<int>(val);
            else if (param == "max_timesteps_per_episode") max_timesteps_per_episode = static_cast<int>(val);
            else if (param == "n_updates_per_iteration") n_updates_per_iteration = static_cast<int>(val);
            else if (param == "lr") lr = val;
            else if (param == "gamma") gamma = val;
            else if (param == "lambda") lambda = val;
            else if (param == "clip") clip = val;
            else if (param == "render") render = (val != 0.0);
            else if (param == "render_every_i") render_every_i = static_cast<int>(val);
            else if (param == "save_freq") save_freq = static_cast<int>(val);
            else if (param == "seed") seed = static_cast<int>(val);
            else if (param == "num_minibatches") num_minibatches = static_cast<int>(val);
            else if (param == "ent_coef") ent_coef = static_cast<int>(val);
            else if (param == "max_grad_norm") max_grad_norm = static_cast<int>(val);
            else if (param == "target_kl") target_kl = static_cast<int>(val);
            //Add more parameters here if needed
        }

        //Sets the seed if specified
        if (seed.has_value()) {
            //Set the seed
            torch::manual_seed(seed.value());
            cout << "Successfully set seed to " << seed.value() << endl;
        }
    }

    void _log_train() {
        try {
            long long prev_delta_t = std::get<long long>(logger["delta_t"]);
            logger["delta_t"] = chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now().time_since_epoch()).count();
            float delta_t_sec = (std::get<long long>(logger["delta_t"]) - prev_delta_t) / 1e9;

            stringstream delta_t_ss;
            delta_t_ss << fixed << setprecision(2) << delta_t_sec;
            string delta_t = delta_t_ss.str();

            int t_so_far = std::get<int>(logger["t_so_far"]);
            int i_so_far = std::get<int>(logger["i_so_far"]);

            vector<int> batch_lengths = std::get<vector<int>>(logger["batch_lengths"]);
            float avg_ep_lens = 0.0;
            if (!batch_lengths.empty()) {
                avg_ep_lens = accumulate(batch_lengths.begin(), batch_lengths.end(), 0.0) / batch_lengths.size();
            }

            vector<vector<float>> batch_rewards = std::get<vector<vector<float>>>(logger["batch_rewards"]);
            float avg_ep_rews = 0.0;
            if (!batch_rewards.empty()) {
                float sum_rews = 0.0;
                int count = 0;
                for (const auto& ep_rews : batch_rewards) {
                    sum_rews += accumulate(ep_rews.begin(), ep_rews.end(), 0.0);
                    count++;
                }
                avg_ep_rews = sum_rews / count;
            }

            auto actor_loss = std::get<torch::Tensor>(logger["actor_loss"]);
            float avg_actor_loss = 0.0;
            if (actor_loss.numel() > 0) {
                if (actor_loss.dim() == 0) {
                    //scalar tensor
                    avg_actor_loss = actor_loss.item<float>();
                }
                else {
                    float sum_loss = 0.0;
                    for (int i = 0; i < actor_loss.size(0); ++i) {
                        sum_loss += actor_loss[i].item<float>();
                    }
                    avg_actor_loss = sum_loss / actor_loss.size(0);
                }
            }

            stringstream avg_ep_lens_ss, avg_ep_rews_ss, avg_actor_loss_ss;
            avg_ep_lens_ss << fixed << setprecision(2) << avg_ep_lens;
            avg_ep_rews_ss << fixed << setprecision(2) << avg_ep_rews;
            avg_actor_loss_ss << fixed << setprecision(5) << avg_actor_loss;

            cout << endl;
            cout << "-------------------- Iteration #" << i_so_far << " --------------------" << endl;
            cout << "Average Episodic Length: " << avg_ep_lens_ss.str() << endl;
            cout << "Average Episodic Return: " << avg_ep_rews_ss.str() << endl;
            cout << "Average Loss: " << avg_actor_loss_ss.str() << endl;
            cout << "Timesteps So Far: " << t_so_far << endl;
            cout << "Iteration took: " << delta_t << " secs" << endl;
            cout << "------------------------------------------------------" << endl;
            cout << endl;

            logger["batch_lengths"] = vector<int>{};
            logger["batch_rewards"] = vector<vector<float>>{};
            logger["actor_loss"] = torch::Tensor();
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in _log_summary: " << e.what() << std::endl;
        }
    }

    torch::Tensor compute_rtgs(const vector<vector<float>>& batch_rewards) {
        //The rewards-to-go (rtg) per episode per batch to return.
        //The shape will be (num timesteps per episode)
        vector<float> batch_rtgs;

        //Iterate through each episode
        for (auto it = batch_rewards.rbegin(); it != batch_rewards.rend(); ++it) {
            const auto& ep_rews = *it;
            float discounted_reward = 0; //The discounted reward so far

            //Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            //discounted return (think about why it would be harder starting from the beginning)
            for (auto rit = ep_rews.rbegin(); rit != ep_rews.rend(); ++rit) {
                discounted_reward = *rit + discounted_reward * gamma;
                batch_rtgs.insert(batch_rtgs.begin(), discounted_reward);
            }
        }

        //Convert the rewards-to-go into a tensor
        return torch::tensor(batch_rtgs, torch::kFloat).to(mDevice);
    }

    pair<torch::Tensor, torch::Tensor> get_action(const torch::Tensor& obs_tensor) {
        //Query the actor network for a mean action
        torch::Tensor mean = actor->forward(obs_tensor);

        //Create a distribution with the mean action and std from the covariance matrix
        auto dist = MultivariateNormal(mean, cov_mat);

        //Sample an action from the distribution
        torch::Tensor action_tensor = dist.sample();

        //Calculate the log probability for that action
        torch::Tensor log_prob_tensor = dist.log_prob(action_tensor);

        print_tensor_inline("obs_tensor", obs_tensor);
        print_tensor_inline("mean", mean);
        print_tensor_inline("action_tensor", action_tensor);
        print_tensor_inline("log_prob_tensor", log_prob_tensor);

        return { action_tensor, log_prob_tensor.detach() };
    }

    tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(const torch::Tensor& batch_obs, const torch::Tensor& batch_acts) {
        //Estimate the values of each observation, and the log probs of
        //each action in the most recent batch with the most recent
        //iteration of the actor network. Should be called from learn.

        //Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        torch::Tensor V = critic->forward(batch_obs).squeeze();

        //Calculate the log probabilities of batch actions using most recent actor network.
        //This segment of code is similar to that in get_action()
        torch::Tensor mean = actor->forward(batch_obs);
        MultivariateNormal dist(mean, cov_mat);
        torch::Tensor log_probs = (dist.log_prob(batch_acts));

        //Return the value vector V of each observation in the batch
        //and log probabilities log_probs of each action in the batch
        //and entropy
        return { V, log_probs, dist.entropy()};
    }

    tuple<torch::Tensor, torch::Tensor, torch::Tensor, vector<vector<float>>, vector<int>, vector<vector<torch::Tensor>>, vector<vector<bool>>> rollout_train() {
        //Batch data. For more details, check function header.
        vector<torch::Tensor> batch_obs_vec;
        vector<torch::Tensor> batch_acts_vec;
        vector<torch::Tensor> batch_log_probs_vec;
        vector<vector<float>> batch_rewards;
        vector<vector<torch::Tensor>> batch_vals;
        vector<vector<bool>> batch_dones;
        vector<int> batch_lengths_vec;

        //Episodic data. Keeps track of rewards per episode, will get cleared
        //upon each new episode
        vector<float> ep_rews;
        vector<torch::Tensor> ep_vals;
        vector<bool> ep_dones;

        int t = 0; //Keeps track of how many timesteps we've run so far this batch

        //Keep simulating until we've run more than or equal to specified timesteps per batch
        while (t < timesteps_per_batch) {
            ep_rews.clear(); //rewards collected per episode
            ep_vals.clear(); //rewards collected per episode
            ep_dones.clear(); //rewards collected per episode

            //Reset the environment. Note that obs is short for observation.
            auto [obs_tensor, _] = env.reset();
            bool done = false;

            //Run an episode for a maximum of max_timesteps_per_episode timesteps
            for (int ep_t = 0; ep_t < max_timesteps_per_episode; ++ep_t) {
                //If render is specified, render the environment
                if (render && (std::get<int>(logger["i_so_far"]) % render_every_i == 0) && batch_lengths_vec.empty()) {
                    env.render();
                }

                t += 1; //Increment timesteps ran this batch so far

                //Track observations in this batch
                batch_obs_vec.push_back(obs_tensor);

                //Calculate action and make a step in the env.
                //Note that rew is short for reward.
                auto [action_tensor, log_prob] = get_action(obs_tensor);
                auto val = critic(obs_tensor);

                auto [next_obs, rew, terminated, truncated, __] = env.step(action_tensor);
                print_tensor_inline("log_prob", log_prob);

                //Don't really care about the difference between terminated or truncated in this, so just combine them
                done = terminated || truncated;

                //Track recent reward, action, and action log probability
                ep_rews.push_back(rew);
                ep_dones.push_back(done);
                ep_vals.push_back(val.flatten());
                batch_acts_vec.push_back(action_tensor);
                batch_log_probs_vec.push_back(log_prob);

                obs_tensor = next_obs;

                //If the environment tells us the episode is terminated, break
                if (done) {
                    break;
                }
            }

            //Track episodic lengths and rewards
            batch_lengths_vec.push_back(ep_rews.size());
            batch_rewards.push_back(ep_rews);
            batch_vals.push_back(ep_vals);
            batch_dones.push_back(ep_dones);
        }

        //Reshape data as tensors in the shape specified in function description, before returning
        torch::Tensor batch_obs = torch::stack(batch_obs_vec).to(torch::kFloat);
        torch::Tensor batch_acts = torch::stack(batch_acts_vec).to(torch::kFloat);
        torch::Tensor batch_log_probs = torch::stack(batch_log_probs_vec).to(torch::kFloat);

        //Log the episodic returns and episodic lengths in this batch.
        logger["batch_rewards"] = batch_rewards;
        logger["batch_lengths"] = batch_lengths_vec;

        return { batch_obs, batch_acts, batch_log_probs, batch_rewards, batch_lengths_vec, batch_vals, batch_dones };
    }

    torch::Tensor calculate_gae(
        const std::vector<std::vector<float>>& rewards,
        const std::vector<std::vector<torch::Tensor>>& values,
        const std::vector<std::vector<bool>>& dones,
        float gamma,
        float lambda
    ) {
        std::vector<float> batch_advantages;

        for (size_t i = 0; i < rewards.size(); ++i) {
            const auto& ep_rews = rewards[i];
            const auto& ep_vals = values[i];
            const auto& ep_dones = dones[i];

            std::vector<float> advantages(ep_rews.size());
            float last_advantage = 0.0f;

            for (int t = static_cast<int>(ep_rews.size()) - 1; t >= 0; --t) {
                torch::Tensor delta;
                if (t + 1 < ep_rews.size()) {
                    delta = torch::tensor(ep_rews[t]) +
                        gamma * ep_vals[t + 1] * (1 - static_cast<float>(ep_dones[t + 1])) -
                        ep_vals[t];
                }
                else {
                    delta = torch::tensor(ep_rews[t]) - ep_vals[t];
                }

                torch::Tensor advantage = delta + gamma * lambda * (1 - static_cast<float>(ep_dones[t])) * last_advantage;
                last_advantage = advantage.item<float>();
                advantages[t] = last_advantage;
            }

            batch_advantages.insert(batch_advantages.end(), advantages.begin(), advantages.end());
        }

        return torch::tensor(batch_advantages, torch::dtype(torch::kFloat32)).to(mDevice);
    }

    Env& env;
    int obs_dim;
    int act_dim;
    float lr;

    torch::Tensor cov_var;
    torch::Tensor cov_mat;
    
    FeedForwardNN actor = nullptr;
    FeedForwardNN critic = nullptr;
    torch::Device mDevice;

    unique_ptr<torch::optim::Adam> actor_optim;
    unique_ptr<torch::optim::Adam> critic_optim;

    using LoggerValue = variant<
        string,
        float,
        int,
        vector<float>,
        vector<int>,
        vector<torch::Tensor>,
        vector<vector<float>>,
        long long,
        torch::Tensor
    >;

    unordered_map<string, LoggerValue> logger;
    vector<int> batch_lengths;
    vector<float> batch_rewards;

    //Algorithm hyperparameters
    int timesteps_per_batch;
    int max_timesteps_per_episode;
    int n_updates_per_iteration;
    float gamma;
    float lambda;
    float clip;
    int num_minibatches;
    float ent_coef;
    //Miscellaneous parameters
    bool render;
    int render_every_i;
    int save_freq;
    optional<int> seed;
    float max_grad_norm;
    float target_kl;
};

class PPO_Eval {
public:
    PPO_Eval(Env& env, torch::Device& device, string actor_model = "")
        : env(env){

        if (actor_model.empty()) {
            cerr << "No actor model file. Exiting." << endl;
            exit(0);
        }


        obs_dim = env.observation_space().shape[0];
        act_dim = env.action_space().shape[0];

        actor = FeedForwardNN(obs_dim, act_dim, device);

        torch::load(actor, actor_model);

        cov_mat = torch::full({ act_dim }, 0.5).to(device).diag();
    }

    void eval_policy(bool render = false) {
        int ep_num = 0;

        while (true) {
            auto [obs_tensor, _] = env.reset();
            bool done = false;

            int t = 0;
            float ep_len = 0.0f;
            float ep_ret = 0.0f;

            while (!done) {
                t++;

                if (render) {
                    env.render();
                }
                auto [action_tensor, log_prob] = get_action(obs_tensor);
                auto [next_obs, rew, terminated, truncated, __] = env.step(action_tensor);
                done = terminated || truncated;

                ep_ret += rew;
                obs_tensor = next_obs;
            }

            ep_len = static_cast<float>(t);

            log_eval(ep_len, ep_ret, ep_num);
            ep_num++;
        }
    }

private:

    pair<torch::Tensor, torch::Tensor> get_action(const torch::Tensor& obs_tensor) {
        //Query the actor network for a mean action
        torch::Tensor mean = actor->forward(obs_tensor);

        //Create a distribution with the mean action and std from the covariance matrix
        auto dist = MultivariateNormal(mean, cov_mat);

        //Sample an action from the distribution
        torch::Tensor action_tensor = dist.sample();

        //Calculate the log probability for that action
        torch::Tensor log_prob_tensor = dist.log_prob(action_tensor);

        print_tensor_inline("obs_tensor", obs_tensor);
        print_tensor_inline("mean", mean);
        print_tensor_inline("action_tensor", action_tensor);
        print_tensor_inline("log_prob_tensor", log_prob_tensor);

        return { action_tensor, log_prob_tensor.detach() };
    }

    void log_eval(float ep_len, float ep_ret, int ep_num) {
        //Round decimals for nicer output
        ep_len = std::round(ep_len * 100.0f) / 100.0f;
        ep_ret = std::round(ep_ret * 100.0f) / 100.0f;

        std::cout << std::endl;
        std::cout << "-------------------- Episode #" << ep_num << " --------------------" << std::endl;
        std::cout << "Episodic Length: " << ep_len << std::endl;
        std::cout << "Episodic Return: " << ep_ret << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << std::endl;
        std::cout.flush();
    }

    FeedForwardNN actor = nullptr;
    Env& env;
    int obs_dim;
    int act_dim;
    torch::Tensor cov_mat;
};

void train(
    Env& env,
    const unordered_map<string, float>& hyperparameters,
    torch::Device& device,
    const string& actor_model,
    const string& critic_model
) {
    cout << "Training" << endl;

    PPO model(env, hyperparameters, device, actor_model, critic_model);

    //Train PPO model for a large number of timesteps
    model.learn(200000000);
}

void eval(Env& env, torch::Device& device, const string& actor_model) {
    cout << "Testing " << actor_model << endl;

    PPO_Eval model(env, device, actor_model);

    model.eval_policy();
}

int main(int argc, char* argv[]) {
    //Hyperparameters for PPO (can be customized here)
    unordered_map<string, float> hyperparameters = {
        {"timesteps_per_batch", 2048},
        {"max_timesteps_per_episode", 200},
        {"gamma", 0.95},
        {"lambda", 0.98},
        {"n_updates_per_iteration", 10},
        {"lr", 3e-4},
        {"clip", 0.2},
        {"render", 0},           //Using 1 for true
        {"render_every_i", 10},
        {"num_minibatches", 10},
        {"ent_coef", 0.9 },
        {"max_grad_norm", 0.5 },
        {"target_kl", 0.02 }
    };


    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

    torch::Device device = torch::Device(torch::kCPU);

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

        device = torch::Device(torch::kCUDA, 0);
    }
    else {
        std::cout << "CUDA is NOT available. CPU will be used.\n";
    }

    try {
        CartPoleEnv env(device);
        if (true) {
            train(env, hyperparameters, device, "", "");
        }
        else {
            eval(env, device, "./models/ppo_actor.pt"); //only load the actor model
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred." << std::endl;
    }

    return 0;
}
