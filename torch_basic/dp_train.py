


from dataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
discriminator = nn.DataParallel(Discriminator(from_rgb_activate=not args.).cuda()
net.to(device)