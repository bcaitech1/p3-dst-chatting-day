
if args.do_eval_best_acc:
    output_model_file = os.path.join(args.output_dir, "acc.best")
else:
    output_model_file = os.path.join(args.output_dir, "loss.best")
model = BeliefTracker(args, num_labels, device)

if args.local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

ptr_model = torch.load(output_model_file)

if n_gpu == 1:
    state = model.state_dict()
    state.update(ptr_model)
    model.load_state_dict(state)
else:
    print("Evaluate using only one device!")
    model.module.load_state_dict(ptr_model)
# in the case that slot and values are different between the training and evaluation

model.to(device)

# Evaluation
if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

    eval_examples = chan_dst_test()
    #all_input_ids, all_input_len, all_label_ids, all_update = convert_examples_to_features(
    #    eval_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
    #all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device)
    #all_update = all_update.to(device)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    #eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids, all_update)
    eval_dataset = SUMBTDataset(eval_examples, label_list, tokenizer, max_seq_length=args.max_seq_length, max_turn_length=args.max_turn_length)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: collate_fn(x))

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    eval_update_acc = 0
    eval_loss_slot, eval_acc_slot = None, None
    nb_eval_steps, nb_eval_examples = 0, 0

    accuracies = {'joint7':0, 'slot7':0, 'joint5':0, 'slot5':0, 'joint_rest':0, 'slot_rest':0,
                  'num_turn':0, 'num_slot7':0, 'num_slot5':0, 'num_slot_rest':0}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_len, label_ids, update = batch
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(0)
            input_len = input_len.unsqueeze(0)
            label_ids = label_ids.unsuqeeze(0)
            update = update.unsqueeze(0)

        with torch.no_grad():
            if n_gpu == 1:
                loss, loss_slot, acc, acc_slot, pred_slot, tup = model(input_ids, input_len, label_ids, update, n_gpu)
            else:
                loss, _, acc, acc_slot, pred_slot, tup_1, tup_2, tup_3 = model(input_ids, input_len, label_ids, update, n_gpu)
                tup = (tup_1.mean(), tup_2.mean(), tup_3.mean())
                nbatch = label_ids.size(0)
                nslot = pred_slot.size(3)
                pred_slot = pred_slot.view(nbatch, -1, nslot)

        accuracies = eval_all_accs(pred_slot, label_ids, accuracies)

        nb_eval_ex = (label_ids[:,:,0].view(-1) != -1).sum().item()
        nb_eval_examples += nb_eval_ex
        nb_eval_steps += 1
        eval_update_acc += tup[2] * nb_eval_ex

        if n_gpu == 1:
            eval_loss += loss.item() * nb_eval_ex
            eval_accuracy += acc.item() * nb_eval_ex
            if eval_loss_slot is None:
                eval_loss_slot = [ l * nb_eval_ex for l in loss_slot]
                eval_acc_slot = acc_slot * nb_eval_ex
            else:
                for i, l in enumerate(loss_slot):
                    eval_loss_slot[i] = eval_loss_slot[i] + l * nb_eval_ex
                eval_acc_slot += acc_slot * nb_eval_ex
        else:
            eval_loss += sum(loss) * nb_eval_ex
            eval_accuracy += sum(acc) * nb_eval_ex

    eval_update_acc = eval_update_acc / nb_eval_examples
    eval_loss = eval_loss / nb_eval_examples
    eval_accuracy = eval_accuracy / nb_eval_examples
    if n_gpu == 1:
        eval_acc_slot = eval_acc_slot / nb_eval_examples

    loss = tr_loss / nb_tr_steps if args.do_train else None

    if n_gpu == 1:
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'loss': loss,
                  'eval_loss_slot':'\t'.join([ str(val/ nb_eval_examples) for val in eval_loss_slot]),
                  'eval_acc_slot':'\t'.join([ str((val).item()) for val in eval_acc_slot])
                    }
    else:
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'loss': loss
                  }

    out_file_name = 'eval_results'
    if args.target_slot=='all':
        out_file_name += '_all'
    output_eval_file = os.path.join(args.output_dir, "%s.txt" % out_file_name)

    if n_gpu == 1:
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
