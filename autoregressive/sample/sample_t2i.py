import pdb

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import time
import argparse
import sys
sys.path.append('/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen')
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")
    ##################################################################
    # import numpy as np
    # from PIL import Image
    # import torch.nn.functional as F
    # img_path = "/home/ldap-users/Share/Corpora/Spoken_Image/Flickr8k_SAS/Data_for_SAS/images/997722733_0cb5439472.jpg"
    # out_path = "/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/997722733_0cb5439472_vqgan.png"
    # input_size = args.image_size
    # img = Image.open(img_path).convert("RGB")
    #
    # # preprocess
    # size_org = img.size
    # img = img.resize((input_size, input_size))
    # img = np.array(img) / 255.
    # x = 2.0 * img - 1.0 # x value is between [-1, 1]
    # x = torch.tensor(x)
    # x = x.unsqueeze(dim=0)
    # x = torch.einsum('nhwc->nchw', x)
    # x_input = x.float().to("cuda")
    # # inference
    # with torch.no_grad():
    #     latent, _, [_, _, indices] = vq_model.encode(x_input)
    #     output = vq_model.decode_code(indices, latent.shape) # output value is between [-1, 1]
    #
    # # postprocess
    # output = F.interpolate(output, size=[size_org[1], size_org[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    # sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    # # save
    # Image.fromarray(sample).save(out_path)
    # print("Reconstructed image is saved to {}".format(out_path))
    ##################################################################

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    # prompts = [
    #     "A kangaroo in an orange hoodie and blue sunglasses stood proudly on the grass in front of the Sydney Opera House. He held a big sign on his chest that said “Welcome Friends!” Every morning, the friendly kangaroo named Kip hopped to the same spot to greet visitors from around the world. Sometimes, he even handed out tiny paper stars to kids who smiled back.",
    #     "A shiny blue Porsche sat quietly in front of a yellow brick wall. The little car loved this sunny corner—it was his secret thinking spot. He dreamed of racing clouds in the sky and exploring faraway lands where walls were painted in every color.",
    #     "In the middle of a quiet forest, an astronaut rode a gentle brown horse. They stopped by a river filled with water lilies. The astronaut gently whispered to his horse, “Let’s see what’s across the river.” Together, they crossed slowly, water lilies swaying beside them like soft green dancers.",
    #     "On a table sat a map of the United States made entirely out of sushi. Each tiny roll made a different state! “Each state has a flavor!” laughed Chef Panda, as he served the magical sushi map. When you tasted California, it sparkled like the ocean—and Texas tasted like a cowboy’s campfire stew.",
    #     "A small fox wearing a yellow raincoat stands at the edge of a puddle in a quiet forest. Raindrops fall gently from the trees above, making soft ripples in the water. The fox holds a tiny paper boat in one paw and stares at the stream as if it's a great river. Behind him, glowing mushrooms light the path, and somewhere deep in the woods, something faintly twinkles.",
    #     "In the middle of a mountain meadow, a bear sits quietly, planting tiny seeds in a circle around him. He wears round glasses and a patchy gardener’s hat. As the sun rises, the seeds bloom instantly into colorful flowers—each one with a face that smiles and hums softly. Butterflies hover near him, and the bear carefully writes in a little notebook made of leaves.",
    #     "A young penguin named Penny climbs into a striped hot air balloon with a backpack full of maps and cookies. The balloon lifts gently from an iceberg and floats above frozen mountains and sleepy polar bears. Inside her basket is a telescope, a compass, and a tiny radio that only picks up whale songs. She waves goodbye to a puffin who gave her a lucky fish scale for the journey.",
    #     "Near a quiet lake at dawn, a green frog with a painter’s smock and messy hands dips a brush into a puddle of sunrise light. With each stroke in the air, a rainbow ribbon appears and dances across the sky. Birds fly through the ribbons and sparkle for a moment. The frog sighs happily, painting the sky one stripe at a time, while a turtle watches from a rock with a cup of tea.",
    #     "Milo the gray tabby cat wears a little apron and opens his tiny bakery just as the moon rises. His kitchen is lit by lanterns made from fireflies, and the smell of cinnamon fills the night air. Animals from the forest line up under the stars to buy glowing moon pies, cloud croissants, and warm starlight muffins. Milo hums softly as he dusts sugar onto each treat and wraps them in dream-paper.",
    #     "Deep beneath the ocean waves, a clever turtle named Timmy builds a time machine out of coral, seashells, and a glowing jellyfish core. His control panel is made of starfish buttons, and the seat is a soft clam shell. As he flips the final switch, bubbles swirl around him, and he’s transported to an underwater city from a thousand years ago—where sea dragons wear crowns and octopuses paint on pearl walls.",
    #     "High in a pine tree, hidden behind a sliding branch door, lives the Midnight Library—run by a wise old owl named Orla. Each book is bound in moonlight and only opens when the reader is truly curious. Tonight, a nervous squirrel arrives with a question so big, it glows from his eyes. Orla flutters down from her perch, selects a book that hums softly, and whispers, “This story knows your heart.",
    #     "On top of a grassy hill that touches the sky, Luna the llama wears a scarf made of morning mist and watches the annual Cloud Parade float by. Each cloud has a shape and personality: the lion cloud roars softly, the teacup cloud spins slowly, and the dancing dolphin cloud makes her giggle. Luna waves her hooves to the crowd above as a shy raindrop slips down her nose and makes her smile.",
    #     "Luna the young wolf followed a floating lantern to a pond. Inside was a flickering star. She brought it home. It glowed when someone felt lonely. The forest called it the Kindness Light.",
    #     "Nina the squirrel carved tiny stories into acorns. Animals borrowed them from her tree library. One snowy night, she found a thank-you note in a snowflake. “Even stories grow roots,” she smiled",
    #     "Freddy the frog found a magical paintbrush. He painted glowing shapes in the sky—stars, ribbons, sparkles. Forest animals watched in awe. “Every day needs a little magic,” Freddy said, hopping away.",
    #     "Ziggy the zebra woke to find his stripes turned rainbow. Other zebras cheered, and a chameleon said, “Nice new look!” Ziggy danced under the sun and painted his colorful self on rocks."
    # ]
    prompts = [
        "in the middle of a mountain meadow, a bear sits quietly, planting tiny seeds in a circle around him.",
    ]
    print(prompts)

    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)# caption_embs: [B, T, D], emb_masks: [B, T]
    # caption_embs = torch.rand_like(caption_embs)  # for testing
    # emb_masks = torch.ones_like(emb_masks)  # for testing
    if not args.no_left_padding:
        print(f"processing left-padding...")    
        # a naive way to implement left-padding
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
    else:
        new_caption_embs, new_emb_masks = caption_embs, emb_masks
    c_indices = new_caption_embs * new_emb_masks[:,:, None]
    c_emb_masks = new_emb_masks
    qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, latent_size ** 2, 
        c_emb_masks, 
        cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )
    sampling_time = time.time() - t1
    print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    save_image(samples, "sample_{}.png".format(args.gpt_type), nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{args.gpt_type}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)
