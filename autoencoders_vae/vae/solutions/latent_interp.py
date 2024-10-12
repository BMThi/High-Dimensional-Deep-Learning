def interpolate_latent_space(x1, x2, num_steps=10):
    vae.eval()
    with torch.no_grad():
        x1, x2 = x1.to(device), x2.to(device)
        mu1, logvar1 = vae.encode(x1.view(1, 1, 28, 28))
        mu2, logvar2 = vae.encode(x2.view(1, 1, 28, 28))

        z1 = vae.sample(mu1, logvar1)
        z2 = vae.sample(mu2, logvar2)

        z = torch.stack([z1 * (1 - t) + z2 * t for t in torch.linspace(0, 1, num_steps)])
        samples = vae.decode(z).cpu()
        samples = samples.view(num_steps, 1, 28, 28)

        fig, ax = plt.subplots(1, num_steps, figsize=(15, 2))
        for i in range(num_steps):
            ax[i].imshow(samples[i].squeeze(0), cmap='gray')
            ax[i].axis('off')
        plt.show()

x1, x2 = test_dataset[3][0], test_dataset[2][0]
interpolate_latent_space(x1, x2)