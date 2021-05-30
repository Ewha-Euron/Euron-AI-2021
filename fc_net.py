
            if not idx == self.num_layers - 1 and self.use_dropout:
                out, cache_do = dropout_forward(out,self.dropout_param)
                cache = (cache, cache_do)

            cachelist.append(cache)

        scores = out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        dout = dscores
        for idx in reversed(range(self.num_layers)):
            cache = cachelist[idx]
            if idx == self.num_layers - 1:
                dout, dw, db = affine_backward(dout, cache)
            else:
                if self.use_dropout:
                    cache, cache_do = cache
                    dout = dropout_backward(dout, cache_do)

                if self.use_batchnorm:
                    dout, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, cache)
                    grads["gamma%d" % (idx + 1)] = dgamma
                    grads["beta%d" % (idx + 1)] = dbeta
                else:
                    dout, dw, db = affine_relu_backward(dout, cache)

            grads["W%d" % (idx + 1)] = dw
            grads["b%d" % (idx + 1)] = db

        for idx in range(self.num_layers):
            w = "W%d" % (idx + 1)
            if self.reg > 0:
                loss += 0.5 * self.reg * (self.params[w] ** 2).sum()
                grads[w] += self.reg * self.params[w]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
