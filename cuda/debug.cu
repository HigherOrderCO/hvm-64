#include "debug.cuh"

__device__ __host__ void stop(const char* tag) {
  //printf(tag);
  //printf("\n");
}

__device__ __host__ bool dbug(u32* K, const char* tag) {
  *K += 1;
  if (*K > 1000000) {
    stop(tag);
    return false;
  }
  return true;
}

__host__ const char* show_ptr(Ptr ptr, u32 slot) {
  static char buffer[8][12];
  if (ptr == 0) {
    strcpy(buffer[slot], "           ");
    return buffer[slot];
  } else if (ptr == BSY) {
    strcpy(buffer[slot], "[.........]");
    return buffer[slot];
  } else {
    const char* tag_str = NULL;
    switch (tag(ptr)) {
      case VR1: tag_str = "VR1"; break;
      case VR2: tag_str = "VR2"; break;
      case NIL: tag_str = "NIL"; break;
      case REF: tag_str = "REF"; break;
      case NUM: tag_str = "NUM"; break;
      case ERA: tag_str = "ERA"; break;
      case VRR: tag_str = "VRR"; break;
      case RDR: tag_str = "RDR"; break;
      case RD1: tag_str = "RD1"; break;
      case RD2: tag_str = "RD2"; break;
      case CON: tag_str = "CON"; break;
      case DUP: tag_str = "DUP"; break;
      case TRI: tag_str = "TRI"; break;
      case QUA: tag_str = "QUA"; break;
      case QUI: tag_str = "QUI"; break;
      case SEX: tag_str = "SEX"; break;
    }
    snprintf(buffer[slot], sizeof(buffer[slot]), "%s:%07X", tag_str, val(ptr));
    return buffer[slot];
  }
}

// Prints a net in hexadecimal, limited to a given size
void print_net(Net* net) {
  printf("Root:\n");
  printf("- %s\n", show_ptr(net->root,0));
  printf("Bags:\n");
  for (u32 i = 0; i < BAGS_SIZE; ++i) {
    Ptr a = net->bags[i].lft;
    Ptr b = net->bags[i].rgt;
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
    }
  }
  printf("Node:\n");
  for (u32 i = 0; i < NODE_SIZE; ++i) {
    Ptr a = net->node[i].ports[P1];
    Ptr b = net->node[i].ports[P2];
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
    }
  }
  printf("BLen: %u\n", net->blen);
  printf("Rwts: %u\n", net->rwts);
  //printf("GTMP: ");
  //for (u32 i = 0; i < GTMP_SIZE; ++i) {
    //printf("%d ", net->gtmp[i]);
  //}
  printf("\n");

}

// Function to insert a new entry into the map
__host__ void map_insert(Map* map, u32 key, u32 val) {
  map->keys[map->size] = key;
  map->vals[map->size] = val;
  map->size++;
}

// Function to lookup an id in the map by key
__host__ u32 map_lookup(Map* map, u32 key) {
  for (u32 i = 0; i < map->size; ++i) {
    if (map->keys[i] == key) {
      return map->vals[i];
    }
  }
  return map->size;
}

// Recursive function to print a term as a tree with unique variable IDs
__host__ void print_tree_go(Net* net, Ptr ptr, Map* var_ids) {
  if (var(ptr)) {
    u32 got = map_lookup(var_ids, ptr);
    if (got == var_ids->size) {
      u32 name = var_ids->size;
      Ptr targ = enter(net, ptr);
      map_insert(var_ids, targ, name);
      printf("x%d", name);
    } else {
      printf("x%d", got);
    }
  } else if (tag(ptr) == ERA) {
    printf("*");
  } else {
    switch (tag(ptr)) {
      case RDR: case RD1: case RD2:
        print_tree_go(net, *target(net, ptr), var_ids);
        break;
      default:
        printf("(%d ", tag(ptr) - CON);
        print_tree_go(net, net->node[val(ptr)].ports[P1], var_ids);
        printf(" ");
        print_tree_go(net, net->node[val(ptr)].ports[P2], var_ids);
        printf(")");
    }
  }
}

__host__ void print_tree(Net* net, Ptr ptr) {
  Map var_ids = { .size = 0 };
  print_tree_go(net, ptr, &var_ids);
  printf("\n");
}
