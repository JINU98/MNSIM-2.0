import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

def test_gpt2_performance(model_name, input_text, num_iterations=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Warm-up run
    model.generate(input_ids, max_length=1000)

    # Measure performance
    total_time = 0
    total_tokens = 0

    for _ in range(num_iterations):
        start_time = time.time()
        output = model.generate(input_ids, max_length=1000)
        end_time = time.time()

        tokens_generated = output.shape[1] - input_ids.shape[1]
        total_tokens += tokens_generated
        total_time += end_time - start_time

    avg_tokens_per_second = total_tokens / total_time
    return avg_tokens_per_second

# List of GPT-2 model sizes
model_sizes = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

# Input text for testing
input_text = '''

In recent years, the rapid growth of technology and the increasing reliance on high-performance computing (HPC) have brought energy efficiency to the forefront of technological discussions. As industries and research organizations increasingly depend on massive computational resources to handle tasks such as artificial intelligence (AI) training, scientific simulations, big data processing, and real-time decision-making, the energy cost associated with these operations has risen significantly. Energy consumption in data centers, supercomputers, and large-scale cloud infrastructure is becoming a limiting factor, both in terms of operational costs and environmental sustainability. As a result, improving energy efficiency in HPC environments is not just an engineering challenge but a global necessity for mitigating the environmental impact of our growing digital world.

### The Importance of Energy Efficiency in HPC

High-performance computing is the backbone of many scientific and industrial applications, from weather forecasting and climate modeling to drug discovery and autonomous vehicle development. These systems often process enormous amounts of data and run complex algorithms requiring large-scale parallel computing. However, the increasing demand for higher computational power comes at the cost of exponentially rising energy consumption.

Energy-efficient computing is essential for several reasons:

1. **Cost-Effectiveness**: Data centers and HPC infrastructures are expensive to maintain, and energy costs represent a significant portion of the operational expenses. Reducing energy consumption directly impacts the bottom line, allowing companies and research institutions to invest in further innovation without increasing operational costs.

2. **Sustainability**: The environmental impact of data centers is substantial, as they consume vast amounts of electricity, often generated from non-renewable sources. The global carbon footprint of data centers is comparable to that of the aviation industry. Improving energy efficiency in HPC can significantly reduce the environmental impact, contributing to the fight against climate change.

3. **Scalability**: As computational demands increase, especially with the growth of artificial intelligence and big data applications, the need for scalable and energy-efficient systems becomes even more crucial. Without advances in energy efficiency, the ability to scale up computing resources will be limited by power consumption and heat dissipation issues.

4. **Performance**: Energy efficiency in computing is often closely linked to performance. Many energy-saving techniques, such as reducing latency and improving hardware utilization, can also lead to faster computation times. Efficient systems that manage energy well can perform complex tasks more quickly and effectively.

### Strategies for Improving Energy Efficiency

To address the challenges of energy consumption in high-performance computing, a variety of strategies and innovations have been developed. These range from hardware-level optimizations to software and algorithmic improvements.

#### 1. **Hardware Accelerators**

One of the most effective ways to improve energy efficiency is through the use of specialized hardware accelerators, such as Graphics Processing Units (GPUs), Tensor Processing Units (TPUs), and custom-designed chips (ASICs). These accelerators are designed to handle specific tasks, such as matrix multiplication in AI models or floating-point operations in scientific simulations, much more efficiently than general-purpose CPUs.

For example, GPUs excel at parallel processing and can perform thousands of operations simultaneously, making them ideal for tasks that involve large-scale data processing. TPUs, developed by Google for machine learning workloads, are optimized for AI training and inference tasks, offering significant energy savings compared to traditional hardware. Similarly, custom ASICs are designed to perform highly specialized tasks with minimal energy consumption.

By offloading specific operations to hardware accelerators, overall system energy consumption can be reduced while maintaining or even improving computational performance.

#### 2. **Parallel Processing and Task Scheduling**

Parallel processing is a cornerstone of high-performance computing, enabling large computational tasks to be divided into smaller ones that can be executed concurrently across multiple processors. Efficient parallelization ensures that all computational resources are utilized effectively, reducing idle time and energy waste.

Task scheduling also plays a critical role in energy efficiency. Advanced scheduling algorithms can dynamically allocate tasks to processors based on their current workload and energy consumption, ensuring that energy-hungry resources are only used when absolutely necessary. Techniques such as dynamic voltage and frequency scaling (DVFS) allow processors to adjust their power consumption based on the current computational load, further improving efficiency.

#### 3. **Software and Algorithmic Optimization**

At the software level, optimizing algorithms to minimize unnecessary computations can have a significant impact on energy efficiency. For instance, many AI and machine learning models involve matrix operations, which are computationally intensive. By using optimized matrix multiplication algorithms, such as Strassen’s algorithm, the number of operations can be reduced, leading to energy savings.






'''

print("Testing GPT-2 performance on available GPUs:")

for model_name in model_sizes:
    try:
        tokens_per_second = test_gpt2_performance(model_name, input_text)
        print(f"{model_name}: {tokens_per_second:.2f} tokens/second")
    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")

print("Performance test completed.")
