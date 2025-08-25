
import torch
import os


def train(args, model, dataloader_train, dataloader_validate=None):
    print('Training model...')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if dataloader_validate is None:
        dataloader_validate = dataloader_train

    best_validate_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        batch_count = 0
        loss_sum = 0
        role_accuracy_sum = 0

        for graphs in dataloader_train:
            optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(True):
                # Extract task embeddings from graphs
                task_embeddings = []
                for i, g in enumerate(graphs):
                    if hasattr(g, 'task_embedding'):
                        task_embeddings.append(g.task_embedding)
                    elif 'task_embedding' in g.graph:
                        task_embeddings.append(g.graph['task_embedding'])
                    else:
                        raise ValueError(f"Graph {i} missing task_embedding. Please ensure all graphs have task_embedding attribute or graph['task_embedding'].")

                # Convert to tensors and ensure all are on device
                task_embedding_tensors = []
                for emb in task_embeddings:
                    if isinstance(emb, torch.Tensor):
                        task_embedding_tensors.append(emb.to(args.device))
                    else:
                        task_embedding_tensors.append(torch.tensor(emb, device=args.device).float())

                # Stack all embeddings
                task_embedding = torch.stack(task_embedding_tensors)

                log_ll, batch_role_accuracy = model(graphs, task_embedding)
                loss = -torch.mean(log_ll)

                role_accuracy_sum += batch_role_accuracy

                if torch.isnan(loss):
                    print('NaN loss detected, skipping batch')
                    continue

                loss.backward()

                if args.clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

            loss_sum += loss.item()
            batch_count += 1

        avg_role_accuracy = role_accuracy_sum / batch_count if batch_count > 0 else 0.0

        epoch_loss = loss_sum / batch_count if batch_count > 0 else float('inf')
        print(
            f'Epoch {epoch + 1}/{args.epochs}, Average Training Loss: {epoch_loss:.4f}, Role Prediction Accuracy: {avg_role_accuracy:.2f}%')

        if dataloader_validate and (epoch + 1) % args.epochs_validate == 0:
            validate_loss, val_role_accuracy = validate(args, model, dataloader_validate)
            print(
                f'Epoch {epoch + 1}/{args.epochs}, Validation Loss: {validate_loss:.4f}, Validation Role Accuracy: {val_role_accuracy:.2f}%')

            if validate_loss < best_validate_loss:
                best_validate_loss = validate_loss
                best_epoch = epoch + 1
                if args.save_model:
                    best_model_path = os.path.join(args.experiment_path, args.model_name)

                    save_content = {
                        'model_state_dict': model.state_dict(),
                        'data_statistics': model.data_statistics,
                        'args': args.__dict__
                    }
                    torch.save(save_content, best_model_path)
                    print(
                        f"Saved best model and statistics to {best_model_path}, Epoch {epoch + 1}, Validation loss: {validate_loss:.4f}")

    print(f'Training completed. Best model at Epoch {best_epoch}, Validation loss: {best_validate_loss:.4f}')


def validate(args, model, dataloader_validate):
    model.eval()
    loss_sum = 0
    batch_count = 0
    role_accuracy_sum = 0

    with torch.no_grad():
        for batch_idx, batch_graphs in enumerate(dataloader_validate):
            if isinstance(batch_graphs, (list, tuple)):
                if len(batch_graphs) > 0:
                    if isinstance(batch_graphs[0], dict) and 'G' in batch_graphs[0]:
                        batch_graphs = [g['G'] for g in batch_graphs]
                    elif isinstance(batch_graphs, tuple) and isinstance(batch_graphs[0], list):
                        batch_graphs = batch_graphs[0]

            if args.dataset == 'mmlu' or args.dataset_name == 'mmlu':
                batch_graphs = [g for g in batch_graphs if g.number_of_nodes() > 0]
                if not batch_graphs:
                    print(f"Warning: All graphs in validation batch {batch_idx} are empty, skipping")
                    continue

            # Extract task embeddings from graphs
            task_embeddings = []
            for i, g in enumerate(batch_graphs):
                if hasattr(g, 'task_embedding'):
                    task_embeddings.append(g.task_embedding)
                elif 'task_embedding' in g.graph:
                    task_embeddings.append(g.graph['task_embedding'])
                else:
                    raise ValueError(f"Graph {i} in validation batch {batch_idx} missing task_embedding. Please ensure all graphs have task_embedding attribute or graph['task_embedding'].")

            # Convert to tensors and ensure all are on device
            task_embedding_tensors = []
            for emb in task_embeddings:
                if isinstance(emb, torch.Tensor):
                    task_embedding_tensors.append(emb.to(args.device))
                else:
                    task_embedding_tensors.append(torch.tensor(emb, device=args.device).float())

            # Stack all embeddings
            task_embedding = torch.stack(task_embedding_tensors)

            log_probs, batch_role_accuracy = model(batch_graphs, task_embedding)
            loss = -log_probs.mean()
            loss_sum += loss.item()
            role_accuracy_sum += batch_role_accuracy
            batch_count += 1

    avg_validate_loss = loss_sum / max(batch_count, 1)
    avg_role_accuracy = role_accuracy_sum / max(batch_count, 1)

    return avg_validate_loss, avg_role_accuracy
