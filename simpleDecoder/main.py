
from config import *
from data import *
from model import *
from train import *

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    #decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words#, decoder_attentions[:di + 1]



def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
def trainIters(encoder, decoder, n_iters, print_every=500, plot_every=100,
               learning_rate=0.01, best_loss=float('Inf')):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.2)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.2)
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)


               
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:

            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            if print_loss_avg < best_loss:
                best_loss = print_loss_avg
                torch.save(encoder, 'encoder.pt')
                torch.save(decoder, 'decoder.pt')
                print("saving model at iteration %d with loss %.4f" % (iter, print_loss_avg))
                evaluateRandomly(encoder, decoder, n=5)
                            

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


if __name__ == '__main__':
    #print(sys.argv)
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    if len(sys.argv) == 1:
        hidden_size = 256
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
        #attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, 1, dropout_p=0.1)
        decoder = DecoderRNN(hidden_size, output_lang.n_words)
    else:
        print('loading models')
        encoder1 = torch.load('encoder.pt')
        decoder = torch.load('decoder.pt')
        
    loss = float('Inf')
    #evaluateRandomly(encoder1, decoder)
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder = decoder.cuda()
    trainIters(encoder1, decoder, 50000, print_every=2500, best_loss=loss)
    evaluateRandomly(encoder1, decoder)
