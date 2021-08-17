import torch
import torch.nn as nn
import torch.optim as optim


def train(training_cfg, generator_cfg, dataloader, generator, discriminator, device):
    criterion = nn.BCELoss()

    optimizerD = optim.Adam(discriminator.parameters(), lr=training_cfg.LEARNING_RATE,
                            betas=(training_cfg.BETA_1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=training_cfg.LEARNING_RATE,
                            betas=(training_cfg.BETA_1, 0.999))

    G_losses = []
    D_losses = []
    real = 1.
    fake = 0.
    iters = 0
    num_epochs = training_cfg.NUM_EPOCHS
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ###########################
            # Train with all-real batch
            discriminator.zero_grad()
            # Create a batch of real data with real labels
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            # Create placeholder label which is populated with real label for now
            label = torch.full((b_size,), real, dtype=torch.float, device=device)
            # Forward pass batch of real data through Discriminator
            output = discriminator(real_data).view(-1)
            # Calculate loss of Discriminator on real data batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            # D_x -> how well discriminator can tell real images as real images (closer to 1 means it can tell real
            # images very well)
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, generator_cfg.LATENT_SIZE, 1, 1, device=device)
            # Generate fake image batch with G
            fake_images = generator(noise)
            # Update placeholder label with all fake labels
            label.fill_(fake)
            # Classify all fake batch with D, detach fake_images tensor from computational graph since it does not
            # need grad
            output = discriminator(fake_images.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            # D_G_z1 -> how well discriminator can tell fake images as fake images (close to 0 mean it can tell fake
            # images very well)
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            generator.zero_grad()
            # For generator we want it to work in a way that it creates convincing images to fool discriminator
            # Therefore, we assign "real" labels to fake images. If the discriminator can tell fake images as
            # fake images, then the generator needs to update its weight since it is not doing a good enough job.
            label.fill_(real)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake_images).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            # D_G_z2 -> how well discriminator can tell fake images as fake images AFTER weight update
            # D_G_z2 should be less than D_G_z1 and should be close to zero if the training is actually working
            # Otherwise, you are not updating weights strong enough or you have forgotten to update weights
            # (close to 0 mean it can tell fake images very well)
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}] [{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item()} Loss_G: {errG.item()} '
                      f'D(x): {D_x} D(G(z))_before: {D_G_z1} D(G(z))_after: {D_G_z2}')

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1
