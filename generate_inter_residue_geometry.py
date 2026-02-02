import os
import math
import torch
import numpy as np
from Bio import SeqIO
from PIL import Image
from my_diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def read_fasta(fasta_file_path):
    result = ""
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        for s in seq_record.seq:
            if s == 'X':
                continue
            result += s
    return result

def image_to_tensor(dis_image, omega_image, theta_image, phi_image, seq_len):
    def image_to_numpy(image):

        image = image.resize((seq_len, int(image.size[1] * seq_len / image.size[0])),Image.NEAREST)
    
        image_np = np.array(image)
        image_np[np.where(image_np>240)]=252

        return image_np
    
    def dis_angle_to_tensor(db, ob, tb, pb):

        db = db.astype(float)
        ob = ob.astype(float)
        db = ((db + db.T)//2).astype(np.uint8)
        ob = ((ob + ob.T)//2).astype(np.uint8)

        db = db * (np.ones([seq_len, seq_len])-np.eye(seq_len))
        db = db + np.eye(seq_len)*252
        ob = ob * (np.ones([seq_len, seq_len])-np.eye(seq_len))
        ob = ob + np.eye(seq_len)*252
        tb = tb * (np.ones([seq_len, seq_len])-np.eye(seq_len))
        tb = tb + np.eye(seq_len)*252
        pb = pb * (np.ones([seq_len, seq_len])-np.eye(seq_len))
        pb = pb + np.eye(seq_len)*252

        db[np.where(db>252)]=252
        ob[np.where(ob>252)]=252
        tb[np.where(tb>252)]=252
        pb[np.where(pb>252)]=252
        ob[np.where(db==252)]=252
        tb[np.where(db==252)]=252
        pb[np.where(db==252)]=252 

        db[np.where(db<7)]=7
        ob[np.where(ob<7)]=7
        tb[np.where(tb<7)]=7
        pb[np.where(pb<14)]=14           
        
        db = torch.from_numpy((db // 7).astype(np.int32))
        ob = torch.from_numpy((ob // 7).astype(np.int32))
        tb = torch.from_numpy((tb // 7).astype(np.int32))
        pb = torch.from_numpy((pb // 14).astype(np.int32))

        i,j = torch.where(db<36)
        mask = torch.zeros((seq_len,seq_len), dtype=db.dtype, device=db.device)
        mask[i,j] = 1.0

        t2d = torch.cat([dist_bin[db.to(torch.long)].unsqueeze(0), 
                        dihedral_bin[ob.to(torch.long)].unsqueeze(0),
                        dihedral_bin[tb.to(torch.long)].unsqueeze(0),
                        angle_bin[pb.to(torch.long)].unsqueeze(0)],dim=0)

        dist = t2d[0]*mask/torch.max(t2d[0])
        dist = torch.clamp(dist, 0.0, 1.0)

        o_sin = torch.sin(t2d[1])*mask
        o_cos = torch.cos(t2d[1])*mask
        t_sin = torch.sin(t2d[2])*mask
        t_cos = torch.cos(t2d[2])*mask
        p_sin = torch.sin(t2d[3])*mask
        p_cos = torch.cos(t2d[3])*mask

        idx = torch.arange(seq_len).long().expand((1, seq_len))
        sep = idx[:,None,:] - idx[:,:,None]
        sep = sep.abs() + torch.eye(seq_len).unsqueeze(0)*999.9
        topk_matrix = torch.zeros((1, seq_len, seq_len))
        cond = torch.logical_or(topk_matrix > 0.0, sep < (seq_len//4))
        b,i,j = torch.where(cond)
        topk_matrix[b,i,j]=1
        topk_matrix = topk_matrix.unsqueeze(1).unsqueeze(-1)

        new_t2d = torch.cat([dist.unsqueeze(-1),
                                o_sin.unsqueeze(-1),
                                t_sin.unsqueeze(-1),
                                p_sin.unsqueeze(-1),
                                o_cos.unsqueeze(-1),
                                t_cos.unsqueeze(-1),
                                p_cos.unsqueeze(-1)],dim=-1)

        new_t2d = topk_matrix*new_t2d.unsqueeze(0).unsqueeze(1)

        return new_t2d

    dist_bin = torch.linspace(2,20,37)
    dihedral_bin = torch.linspace(-1*math.pi,math.pi,37)
    angle_bin = torch.linspace(0,math.pi,19)

    dis_np = image_to_numpy(dis_image)
    omega_np = image_to_numpy(omega_image)
    theta_np = image_to_numpy(theta_image)
    phi_np = image_to_numpy(phi_image)

    t2d_0 = dis_angle_to_tensor(dis_np[...,0], omega_np[...,0], theta_np[...,0], phi_np[...,0])
    t2d_1 = dis_angle_to_tensor(dis_np[...,1], omega_np[...,1], theta_np[...,1], phi_np[...,1])
    t2d_2 = dis_angle_to_tensor(dis_np[...,2], omega_np[...,2], theta_np[...,2], phi_np[...,2])

    return torch.cat([t2d_0, t2d_1, t2d_2],dim=1)

class InterResidueGeometryGenerator:
    def __init__(self, model_id, text_lora_id, unet_lora_dis_id, unet_lora_omega_id, unet_lora_theta_id, unet_lora_phi_id):
        pipe = StableDiffusionPipeline.from_pretrained(model_id, text_lora_id=text_lora_id, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        pipe.tokenizer.model_max_length = 385
        
        pipe.load_lora_weights(unet_lora_dis_id, adapter_name='dis')
        pipe.load_lora_weights(unet_lora_omega_id, adapter_name='omega')
        pipe.load_lora_weights(unet_lora_theta_id, adapter_name='theta')
        pipe.load_lora_weights(unet_lora_phi_id, adapter_name='phi')

        self.pipe = pipe
        self.max_length = 385

    def generate_inter_residue_geometry(self, fasta_path, data_path):
 
        for i, data_name in enumerate(os.listdir(data_path)):

            data = torch.load(os.path.join(data_path, data_name))

            fasta_file_path = os.path.join(fasta_path, data.protein_name+'.fasta')
            fasta_str = read_fasta(fasta_file_path)
            prompt = ' '.join(fasta_str)
            

            input_ids = self.pipe.tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to("cuda")

            if input_ids.shape[-1] > self.max_length:

                negative_ids = self.pipe.tokenizer("", truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
                negative_ids = negative_ids.to("cuda")

                concat_embeds = []
                neg_embeds = []

                for k in range(0, input_ids.shape[-1], self.max_length):
                    
                    temp_input_ids = input_ids[:, k: k + self.max_length]
                    temp_neg_ids = negative_ids[:, k: k + self.max_length]
                    if temp_input_ids.shape[1] < self.max_length:
                        temp_input_ids = torch.nn.functional.pad(temp_input_ids, (1, self.max_length-temp_input_ids.shape[1]-1), mode='constant', value=49407)
                    if temp_neg_ids.shape[1] < self.max_length:    
                        temp_neg_ids = torch.nn.functional.pad(temp_neg_ids, (1, self.max_length-temp_neg_ids.shape[1]-1), mode='constant', value=49407)

                    concat_embeds.append(self.pipe.text_encoder(temp_input_ids)[0])
                    neg_embeds.append(self.pipe.text_encoder(temp_neg_ids)[0])

                prompt_embeds = torch.cat(concat_embeds, dim=1)
                negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

                self.pipe.set_adapters(['text', 'dis'], adapter_weights=[1.0, 1.0])
                image_dis = self.pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=50, guidance_scale=7.5, height=256, width=256).images[0]

                self.pipe.set_adapters(['text', 'omega'], adapter_weights=[1.0, 1.0])
                image_omega = self.pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=50, guidance_scale=7.5, height=256, width=256).images[0]

                self.pipe.set_adapters(['text', 'theta'], adapter_weights=[1.0, 1.0])
                image_theta = self.pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=50, guidance_scale=7.5, height=256, width=256).images[0]

                self.pipe.set_adapters(['text', 'phi'], adapter_weights=[1.0, 1.0])
                image_phi = self.pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=50, guidance_scale=7.5, height=256, width=256).images[0]
            else:
                self.pipe.set_adapters(['text', 'dis'], adapter_weights=[1.0, 1.0])
                image_dis = self.pipe(prompt, num_inference_steps=50, guidance_scale=7.5, height=256, width=256).images[0]

                self.pipe.set_adapters(['text', 'omega'], adapter_weights=[1.0, 1.0])
                image_omega = self.pipe(prompt, num_inference_steps=50, guidance_scale=7.5, height=256, width=256).images[0]

                self.pipe.set_adapters(['text', 'theta'], adapter_weights=[1.0, 1.0])
                image_theta = self.pipe(prompt, num_inference_steps=50, guidance_scale=7.5, height=256, width=256).images[0]

                self.pipe.set_adapters(['text', 'phi'], adapter_weights=[1.0, 1.0])
                image_phi = self.pipe(prompt, num_inference_steps=50, guidance_scale=7.5, height=256, width=256).images[0]

            data.new_t2d = image_to_tensor(image_dis, image_omega, image_theta, image_phi, input_ids.shape[-1]-2)

            torch.save(data, os.path.join(data_path, data_name))

         
